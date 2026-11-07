"""
E8: GBDT + ANN three-model probability stacking (OOF stacking)
=====================================================================================
Companion documents:
  - CBUTransfer/DeepLearning/DeepLearning_Supplementary_Plan.md §9.1 (E8)
  - CBUTransfer/DeepLearning/SupplementaryExperiments/README.md
  - CBUTransfer/DeepLearning/DeepLearning_ANN_Full_Report.md (write-back §6.2.4)

Purpose:
  using the XGBoost-104(tuned) + BiGRU(E1.2 tuned) + FT-Transformer(E2.1 tuned) trio
  softmax probabilities as input, training a LogisticRegression meta-learner on 5-fold OOF probabilities,
  answer "whether the dual-track (GBDT+ANN) complementarity materializes in metrics".

Protocol key points:
  - Data/features/label encoding fully consistent with E4 (104 dims, train 12,537 / val 2,640 / test 2,641);
  - Outer StratifiedKFold(5, shuffle, seed): held-out folds only produce OOF probabilities,
    never participates in training or early stopping;
  - Within each fold, 10% is split off as an inner validation set for early stopping / tree-count selection (XGB early_stopping, torch val Macro-F1);
  - Deep models' StandardScaler is fitted only on the training part within the fold;
  - Each fold model outputs one probability set for test/val each; the 5-fold average (bagged) serves as that model's test probability;
    the meta layer reports two inference metrics: inference on bagged probabilities (bagged) and per-fold meta predictions averaged
    (foldavg, main convention - the inference input distribution matches the OOF training distribution);
  - val is monitoring only, not involved in any selection;
  - Meta-learner LogisticRegression(multinomial, lbfgs, C=1.0) fitted on OOF (C not tuned in pilot).

Reference:
  single model (bagged) / simple average / weighted average (simplex grid on OOF) / OOF stacking /
  candidate reranking (top-20 union, equal-weight log-prob) / oracle upper bound.

Outputs (this directory, pilot adds the _pilot suffix):
  E8_runs{suffix}.csv      metrics per method x split
  E8_folds{suffix}.csv     per-fold training results
  E8_oof{suffix}.npz       OOF/test/val probabilities (float16) and labels
  E8_summary{suffix}.json  machine-readable summary (with criteria/oracle/guardrails)
  E8_report{suffix}.md     human-readable report
  E8_metrics{suffix}.png   method-comparison bar chart (English labels)
  e8_run{suffix}.log       run log

Usage:
  py -B e8_stacking.py --smoke                  # 1 fold / 1 epoch / small tree count, pipeline check
  py -B e8_stacking.py --pilot --seed 42        # single-seed 5-fold x 3-model pilot
  py -B e8_stacking.py --pilot --seed 42 --skip-ftt
"""

import argparse
import csv
import gc
import json
import logging
import math
import os
import sys
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(os.path.abspath(__file__))
SUP_DIR = os.path.abspath(os.path.join(HERE, ".."))
E1_DIR = os.path.join(SUP_DIR, "E1_tuning")
E2_DIR = os.path.join(SUP_DIR, "E2_sota")
for _p in (E1_DIR, E2_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import e2_common as ec  # noqa: E402
from e1_2_bigru import build_model as build_bigru  # noqa: E402
from e2_1_ft_transformer import build_ftt, forward_ftt  # noqa: E402

# ==================== Constants ====================
N_CLASSES = ec.N_CLASS_EXPECT
MODELS = ["XGBoost-104", "BiGRU", "FT-Transformer"]
SLUG = {"XGBoost-104": "xgboost", "BiGRU": "bigru", "FT-Transformer": "ftt"}
REF_MODEL = "XGBoost-104"

# E4 (5 seeds, full training set) test Acc references: XGBoost-104 83.33±0.22 / BiGRU 79.71±0.61 /
# FT-Transformer 78.13±0.63 (percent). Criterion: stacking delta Acc >= +0.5 pp and the paired CI lower bound > 0
E4_REF_ACC = {"XGBoost-104": 0.8333, "BiGRU": 0.7971, "FT-Transformer": 0.7813}
E4_REF_STD = {"XGBoost-104": 0.0022, "BiGRU": 0.0061, "FT-Transformer": 0.0063}
DELTA_PP_REQUIRED = 0.5

# Guardrail reference (XGBoost-104 test): Macro-F1 (test-supported classes only) 0.7155, Top-5 97.30%
GUARDRAIL_MACRO_F1_SUP = 0.7155
GUARDRAIL_TOP5 = 0.9730
GUARDRAIL_TOL_PP = 0.3

# Historical oracle upper bounds (E4 five-seed int16 predictions): XGB+BiGRU 87.02 / XGB+FTT 87.16 / three models 88.68
ORACLE_HIST = {"XGBoost-104+BiGRU": 87.02, "XGBoost-104+FT-Transformer": 87.16,
               "all3": 88.68}

SUFFIX = ""
LOG = None
RUNS_CSV = FOLDS_CSV = OOF_NPZ = SUMMARY_JSON = REPORT_MD = FIG_PNG = LOG_PATH = None

# The 5 paired seeds used for formal conclusions (same as E4)
SEEDS = [42, 123, 777, 2026, 3407]
E4_SEED_CSV = os.path.join(SUP_DIR, "E4_seeds", "seed_repetition_results.csv")

RUN_FIELDS = ["method", "split", "acc", "macro_f1", "macro_f1_support", "weighted_f1",
              "top3", "top5", "top10", "precision_macro", "recall_macro",
              "head_f1", "torso_f1", "tail_f1", "n_samples"]
FOLD_FIELDS = ["fold", "model", "n_train", "n_inner", "n_holdout", "holdout_acc",
               "holdout_macro_f1", "best_epoch", "epochs_run", "n_params",
               "duration_s"]


def log(msg):
    if LOG is not None:
        LOG.info(msg)
    print(msg, flush=True)


# ==================== Argument parsing ====================
def _json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _py(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def load_tuned_params():
    p_xgb = _json(os.path.join(E1_DIR, "best_params.json"))["params"]
    p_bigru = _json(os.path.join(E1_DIR, "best_params_E1_2.json"))["params"]
    ftt_keys = ["d_token", "n_blocks", "n_heads", "attn_dropout", "ffn_mult",
                "ffn_dropout", "resid_dropout", "lr", "weight_decay", "batch_size"]
    h1 = pd.read_csv(os.path.join(E2_DIR, "E2_1_trials_history.csv"))
    row1 = h1[h1["number"] == 21].iloc[0]
    p_ftt = {k: _py(row1[k]) for k in ftt_keys}
    p_ftt["d_token"] = int(p_ftt["d_token"])
    p_ftt["n_blocks"] = int(p_ftt["n_blocks"])
    p_ftt["n_heads"] = int(p_ftt["n_heads"])
    p_ftt["ffn_mult"] = float(p_ftt["ffn_mult"])
    p_ftt["batch_size"] = int(p_ftt["batch_size"])
    return {"XGBoost-104": p_xgb, "BiGRU": p_bigru, "FT-Transformer": p_ftt}


# ==================== Data ====================
def load_raw():
    """Consistent with E4: npz cache labels are pre-encoding raw values, transformed with the same LabelEncoder."""
    z = np.load(ec.CACHE_PATH)
    le = LabelEncoder()
    y_train = le.fit_transform(z["y_train"]).astype(np.int64)
    y_val = le.transform(z["y_val"]).astype(np.int64)
    y_test = le.transform(z["y_test"]).astype(np.int64)
    if len(le.classes_) != N_CLASSES:
        raise RuntimeError(f"class count {len(le.classes_)} != {N_CLASSES}")
    return dict(
        X_train=z["X_train"].astype(np.float32), y_train=y_train,
        X_val=z["X_val"].astype(np.float32), y_val=y_val,
        X_test=z["X_test"].astype(np.float32), y_test=y_test,
        classes=le.classes_,
    )


def make_folds(y, n_folds, seed):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [(tr.astype(np.int64), ho.astype(np.int64))
            for tr, ho in skf.split(np.zeros(len(y)), y)]


def inner_split(idx_tr, y, seed, frac=0.10):
    """Further splits an inner validation set within each fold (for early stopping only). Classes with frequency 1 all stay on the training side,
    avoiding stratify errors; the inner side will not contain classes missing on the training side."""
    counts = np.bincount(y[idx_tr], minlength=N_CLASSES)
    single = np.where(counts == 1)[0]
    is_single = np.isin(y[idx_tr], single)
    pool = idx_tr[~is_single]
    keep = idx_tr[is_single]
    a, b = train_test_split(pool, test_size=frac, stratify=y[pool],
                            random_state=seed)
    return np.sort(np.concatenate([a, keep])), np.sort(b)


# ==================== Training: XGBoost (within fold) ====================
def train_xgb_fold(params, Xtr, ytr, Xin, yin, Xho, Xte, Xva, seed, tag,
                   n_estimators_cap=None):
    ec.set_seed(seed)
    uniq = np.unique(ytr)
    ytr_m = np.searchsorted(uniq, ytr)
    keep_in = np.isin(yin, uniq)
    yin_m = np.searchsorted(uniq, yin[keep_in])
    p = dict(params)
    if n_estimators_cap:
        p["n_estimators"] = int(n_estimators_cap)
    clf = xgb.XGBClassifier(
        tree_method="hist", random_state=seed, n_jobs=-1,
        eval_metric="mlogloss", early_stopping_rounds=50, **p)
    clf.fit(Xtr, ytr_m, eval_set=[(Xin[keep_in], yin_m)], verbose=False)

    def expand(X):
        proba = np.zeros((len(X), N_CLASSES), dtype=np.float32)
        proba[:, uniq[clf.classes_]] = clf.predict_proba(X)
        return proba

    best_iter = int(getattr(clf, "best_iteration", 0)) + 1
    ho_proba, te_proba = expand(Xho), expand(Xte)
    va_proba = expand(Xva)
    del clf
    gc.collect()
    return dict(ho_proba=ho_proba, te_proba=te_proba, va_proba=va_proba,
                best_epoch=best_iter, epochs_run=best_iter, n_params=0)


# ==================== Training: PyTorch (within fold) ====================
def _val_probs_loss(model, forward, X, y, batch=256):
    model.eval()
    outs, total, n = [], 0.0, 0
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.as_tensor(X[i:i + batch], device=ec.DEVICE)
            yb = torch.as_tensor(y[i:i + batch], dtype=torch.long, device=ec.DEVICE)
            logits = forward(model, xb)
            total += float(F.cross_entropy(logits, yb, reduction="sum").item())
            n += len(yb)
            outs.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(outs, axis=0), total / max(n, 1)


def _train_torch_once(build_fn, forward, params, Xtr, ytr, Xin, yin, Xho, Xte,
                      Xva, seed, bs, max_epochs, patience, optimizer_kind,
                      scheduler_kind, label_smoothing, grad_clip,
                      warmup_ratio, tag):
    ec.set_seed(seed)
    train_ds = TensorDataset(torch.as_tensor(Xtr),
                             torch.as_tensor(ytr, dtype=torch.long))
    loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=False)
    model = build_fn(params).to(ec.DEVICE)
    n_params = int(sum(p.numel() for p in model.parameters()))

    Opt = torch.optim.AdamW if optimizer_kind == "adamw" else torch.optim.Adam
    lr = float(params.get("lr", params.get("learning_rate")))
    opt = Opt(model.parameters(), lr=lr,
              weight_decay=float(params.get("weight_decay", 0.0)))

    sched = None
    total_steps = max_epochs * max(1, len(loader))
    if scheduler_kind == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=5)
    elif scheduler_kind == "cosine_warmup":
        warm = max(1, int(float(warmup_ratio or 0.1) * total_steps))

        def lr_lambda(step):
            if step < warm:
                return step / warm
            prog = (step - warm) / max(1, total_steps - warm)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best_f1, best_state, best_epoch, bad = -1.0, None, 0, 0
    epochs_run = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in loader:
            xb = xb.to(ec.DEVICE, non_blocking=True)
            yb = yb.to(ec.DEVICE, non_blocking=True)
            opt.zero_grad()
            loss = F.cross_entropy(forward(model, xb), yb,
                                   label_smoothing=label_smoothing)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            opt.step()
            if scheduler_kind == "cosine_warmup":
                sched.step()

        epochs_run = epoch
        probs, val_loss = _val_probs_loss(model, forward, Xin, yin)
        val_f1 = ec.metrics_from_proba(probs, yin, N_CLASSES)["macro_f1"]
        if scheduler_kind == "plateau":
            sched.step(val_loss)
        if val_f1 > best_f1:
            best_f1, best_epoch, bad = val_f1, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    ho_proba = ec.predict_proba(model, Xho, forward)
    te_proba = ec.predict_proba(model, Xte, forward)
    va_proba = ec.predict_proba(model, Xva, forward)
    log(f"{tag} best_epoch={best_epoch}/{epochs_run} | inner val MacroF1={best_f1:.4f} "
        f"| params={n_params}")
    del model
    if ec.DEVICE.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
    return dict(ho_proba=ho_proba, te_proba=te_proba, va_proba=va_proba,
                best_epoch=int(best_epoch), epochs_run=int(epochs_run),
                n_params=n_params)


def train_torch_fold(spec, params, Xtr, ytr, Xin, yin, Xho, Xte, Xva, seed,
                     max_epochs, patience, tag):
    bs = int(params["batch_size"])
    if ec.DEVICE.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
    for attempt in range(3):
        try:
            return _train_torch_once(
                spec["build"], spec["forward"], params, Xtr, ytr, Xin, yin,
                Xho, Xte, Xva, seed, bs, max_epochs, patience,
                spec["optimizer_kind"], spec["scheduler_kind"],
                spec["label_smoothing"], spec["grad_clip"],
                params.get("warmup_ratio"), tag)
        except torch.OutOfMemoryError:
            if attempt == 2 or bs // 2 < 64:
                raise
            bs = max(64, bs // 2)
            log(f"{tag} | CUDA OOM, batch -> {bs}")
            gc.collect()
            torch.cuda.empty_cache()


TORCH_SPECS = {
    "BiGRU": dict(build=lambda p: build_bigru(p, N_CLASSES),
                  forward=lambda m, x: m(x),
                  max_epochs=ec.MAX_EPOCHS, patience=ec.PATIENCE,
                  optimizer_kind="adamw", scheduler_kind="plateau",
                  label_smoothing=0.1, grad_clip=1.0),
    "FT-Transformer": dict(build=lambda p: build_ftt(p, N_CLASSES),
                           forward=forward_ftt,
                           max_epochs=ec.MAX_EPOCHS, patience=ec.PATIENCE,
                           optimizer_kind="adamw", scheduler_kind="none",
                           label_smoothing=0.0, grad_clip=None),
}


# ==================== Metrics ====================
def macro_f1_support(y, pred):
    present = np.unique(y)
    return float(f1_score(y, pred, labels=present, average="macro", zero_division=0))


def full_metrics(proba, y):
    m = ec.metrics_from_proba(proba, y, N_CLASSES, STRATA)
    m["macro_f1_support"] = macro_f1_support(y, proba.argmax(axis=1))
    m["n_samples"] = int(len(y))
    return m


# ==================== Post-processing: fusion rules ====================
def simple_average(probs, models):
    return np.mean([probs[m] for m in models], axis=0).astype(np.float32)


def search_weights(probs, y, models, step=0.05):
    """Simplex grid search on OOF probabilities for Acc-optimal weights."""
    best_w, best_acc = None, -1.0
    n = len(models)
    grid = np.arange(0.0, 1.0 + 1e-9, step)

    def rec(i, remain, cur):
        nonlocal best_w, best_acc
        if i == n - 1:
            w = np.array(cur + [round(remain, 6)], dtype=np.float64)
            p = sum(w[k] * probs[models[k]] for k in range(n))
            acc = float((p.argmax(axis=1) == y).mean())
            if acc > best_acc:
                best_acc, best_w = acc, w
            return
        for w in grid:
            if w > remain + 1e-9:
                break
            rec(i + 1, remain - w, cur + [float(w)])

    rec(0, 1.0, [])
    return best_w, best_acc


def weighted_average(probs, weights, models):
    return sum(float(weights[k]) * probs[models[k]]
               for k in range(len(models))).astype(np.float32)


def lr_stack(oof, y, test, val, models, C=1.0, max_iter=2000,
             test_folds=None, val_folds=None):
    """Trains LR on OOF probabilities. test_folds/val_folds are {model: [per-fold probabilities]}:
    when provided, additionally output "per-fold meta predictions averaged" (test_foldavg/val_foldavg), so the meta-learner
    infers on an input distribution consistent with OOF (single-fold models), avoiding the mismatch of bagged probabilities (5-fold averaged, higher entropy)."""
    Xo = np.concatenate([oof[m] for m in models], axis=1).astype(np.float32)
    Xt = np.concatenate([test[m] for m in models], axis=1).astype(np.float32)
    Xv = None
    if val is not None and all(val.get(m) is not None for m in models):
        Xv = np.concatenate([val[m] for m in models], axis=1).astype(np.float32)
    clf = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs", tol=1e-4)
    t0 = time.time()
    clf.fit(Xo, y)
    dt = time.time() - t0
    log(f"[stack {len(models)} models] LR fit {dt:.1f}s | iter={clf.n_iter_} | "
        f"OOF(fit) Acc={(clf.predict(Xo) == y).mean():.4f}")

    def expand(p):
        # LR's classes_ may contain fewer than 231 classes (folds/covered samples miss classes); map back to the full column set
        out = np.zeros((len(p), N_CLASSES), dtype=np.float32)
        out[:, clf.classes_] = p
        return out

    return dict(oof=expand(clf.predict_proba(Xo)),
                test=expand(clf.predict_proba(Xt)),
                val=None if Xv is None else expand(clf.predict_proba(Xv)),
                test_foldavg=None if test_folds is None else _foldavg(clf, expand, test_folds, models),
                val_foldavg=None if val_folds is None else _foldavg(clf, expand, val_folds, models),
                n_iter=int(clf.n_iter_[0]) if hasattr(clf.n_iter_, "__len__") else int(clf.n_iter_),
                fit_s=round(dt, 1))


def _foldavg(clf, expand, folds_probs, models):
    """Per fold: make meta predictions with that fold's base-model probabilities, then average the meta outputs over 5 folds."""
    n_fold = len(folds_probs[models[0]])
    outs = []
    for fi in range(n_fold):
        Xf = np.concatenate([folds_probs[m][fi] for m in models],
                            axis=1).astype(np.float32)
        outs.append(expand(clf.predict_proba(Xf)))
    return np.mean(outs, axis=0).astype(np.float32)


def rerank_topk(probs, models, k=20):
    """Within the top-k union of the three models, take argmax after equal-weight log-prob averaging; scores are renormalized back to probability space,
    non-candidate classes set to 0 (only for Top-K ranking and metric computation)."""
    n = len(probs[models[0]])
    out = np.zeros((n, N_CLASSES), dtype=np.float32)
    logs = {m: np.log(np.maximum(probs[m], 1e-12)) for m in models}
    for i in range(n):
        cand = np.unique(np.concatenate(
            [np.argsort(-probs[m][i], kind="stable")[:k] for m in models]))
        s = np.mean([logs[m][i, cand] for m in models], axis=0)
        e = np.exp(s - s.max())
        out[i, cand] = (e / e.sum()).astype(np.float32)
    return out


def oracle_acc(pred_dict, y, models):
    hit = np.zeros(len(y), dtype=bool)
    for m in models:
        hit |= (pred_dict[m] == y)
    return float(hit.mean()), int((~hit).sum())


# ==================== CSV / JSON ====================
def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def append_row(path, row, fields):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)


def metric_row(method, split, m):
    return dict(
        method=method, split=split,
        acc=round(float(m["acc"]), 6),
        macro_f1=round(float(m["macro_f1"]), 6),
        macro_f1_support=round(float(m["macro_f1_support"]), 6),
        weighted_f1=round(float(m["weighted_f1"]), 6),
        top3=round(float(m["top3"]), 6),
        top5=round(float(m["top5"]), 6),
        top10=round(float(m["top10"]), 6),
        precision_macro=round(float(m["precision_macro"]), 6),
        recall_macro=round(float(m["recall_macro"]), 6),
        head_f1=round(float(m.get("head_f1", float("nan"))), 6),
        torso_f1=round(float(m.get("torso_f1", float("nan"))), 6),
        tail_f1=round(float(m.get("tail_f1", float("nan"))), 6),
        n_samples=int(m["n_samples"]),
    )


# ==================== Multi-sub-seed aggregation ====================
def _ci95(d):
    """Paired-difference t-distribution 95% CI (df=n-1)."""
    import scipy.stats as st
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    half = float(st.t.ppf(0.975, n - 1) * sd / math.sqrt(n)) if n > 1 else 0.0
    return mean, sd, mean - half, mean + half


def analyze_multiseed(seeds, tag):
    """Reads each sub-seed's E8_summary{tag}seed{s}.json and performs paired CIs and formal verdicts."""
    paths = [os.path.join(HERE, f"E8_summary_{tag}seed{s}.json") for s in seeds]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("missing sub-seed results: " + ", ".join(os.path.basename(p) for p in missing))
    sums = {s: _json(p) for s, p in zip(seeds, paths)}
    prim = sums[seeds[0]]["verdict"].get("primary_stack_method", "stacking_all3")
    methods = list(sums[seeds[0]]["metrics"].keys())

    e4 = pd.read_csv(E4_SEED_CSV)
    e4x = {int(k): float(v) for k, v in
           e4[e4["model"] == "XGBoost-104"].set_index("seed")["acc"].items()}
    lost = [s for s in seeds if s not in e4x]
    if lost:
        raise KeyError(f"E4 results missing XGBoost-104 for seed {lost}")

    rows = []
    for s in seeds:
        for m in methods:
            mm = sums[s]["metrics"][m]["test"]
            rows.append(dict(method=m, seed=s, acc=round(float(mm["acc"]), 6),
                             macro_f1_support=round(float(mm["macro_f1_support"]), 6),
                             weighted_f1=round(float(mm["weighted_f1"]), 6),
                             top5=round(float(mm["top5"]), 6)))
    seeds_csv = os.path.join(HERE, "E8_seeds_multiseed.csv")
    write_csv(seeds_csv, rows, ["method", "seed", "acc", "macro_f1_support",
                                "weighted_f1", "top5"])

    def vec(m, key):
        return np.array([sums[s]["metrics"][m]["test"][key] for s in seeds], dtype=np.float64)

    agg = {}
    for m in methods:
        agg[m] = {}
        for key in ("acc", "macro_f1_support", "weighted_f1", "top5"):
            v = vec(m, key)
            agg[m][key] = {"mean": round(float(v.mean()), 6),
                           "std": round(float(v.std(ddof=1)), 6),
                           "per_seed": [round(float(x), 6) for x in v]}

    acc_prim = vec(prim, "acc")
    acc_bag = vec("stacking_all3", "acc")
    acc_e4 = np.array([e4x[s] for s in seeds], dtype=np.float64)
    acc_w = vec("weighted_avg", "acc")
    d_e4 = (acc_prim - acc_e4) * 100
    d_w = (acc_prim - acc_w) * 100
    d_bag = (acc_prim - acc_bag) * 100
    m_e4, sd_e4, lo_e4, hi_e4 = _ci95(d_e4)
    m_w, sd_w, lo_w, hi_w = _ci95(d_w)

    f1_prim, top5_prim = vec(prim, "macro_f1_support"), vec(prim, "top5")
    guard = {
        "macro_f1_support_mean": round(float(f1_prim.mean()), 6),
        "macro_f1_support_ref": GUARDRAIL_MACRO_F1_SUP,
        "macro_f1_support_seeds_pass": [int(sum(v >= GUARDRAIL_MACRO_F1_SUP - GUARDRAIL_TOL_PP / 100
                                              for v in f1_prim)), len(seeds)],
        "top5_mean": round(float(top5_prim.mean()), 6),
        "top5_ref": GUARDRAIL_TOP5,
        "top5_seeds_pass": [int(sum(v >= GUARDRAIL_TOP5 for v in top5_prim)), len(seeds)],
    }
    verdict = {
        "criterion": "delta Acc >= +0.5 pp vs E4 XGBoost-104 paired same-seed, and 95% CI lower bound > 0",
        "primary_stack_method": prim,
        "primary_inference": "Per-fold meta prediction then average (foldavg)",
        "delta_vs_e4_ref_pp": {"mean": round(m_e4, 3), "std": round(sd_e4, 3),
                              "ci95_low": round(lo_e4, 3), "ci95_high": round(hi_e4, 3),
                              "per_seed": [round(float(x), 3) for x in d_e4]},
        "delta_vs_weighted_avg_pp": {"mean": round(m_w, 3), "std": round(sd_w, 3),
                                    "ci95_low": round(lo_w, 3), "ci95_high": round(hi_w, 3),
                                    "per_seed": [round(float(x), 3) for x in d_w]},
        "delta_foldavg_minus_bagged_pp": round(float(d_bag.mean()), 3),
        "point_pass": bool(m_e4 >= DELTA_PP_REQUIRED),
        "ci_lower_pass": bool(lo_e4 > 0),
        "formal_pass": bool(m_e4 >= DELTA_PP_REQUIRED and lo_e4 > 0),
        "guardrail": guard,
    }

    summary = {
        "experiment": "E8_multiseed", "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seeds": [int(s) for s in seeds], "primary_method": prim,
        "reference": {"e4_xgb_per_seed": {str(s): round(e4x[s], 6) for s in seeds},
                      "oracle_hist": ORACLE_HIST},
        "aggregate": agg, "verdict": verdict,
        "artifacts": {"seeds_csv": seeds_csv,
                      "summary_json": os.path.join(HERE, "E8_summary_multiseed.json"),
                      "report_md": os.path.join(HERE, "E8_report_multiseed.md")},
    }
    with open(os.path.join(HERE, "E8_summary_multiseed.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = ["# E8 probability stacking (5-seed paired) report", "",
             f"- Seeds: {seeds} | Primary metric: {prim} (foldavg) | Date: {summary['date']}",
             "", "## 1. 5-seed mean +/- std of each method (test)", "",
             "| Method | Acc | Macro-F1(sup) | Weighted-F1 | Top-5 |", "|---|---|---|---|---|"]
    order = [n for n in methods if n in (
        "XGBoost-104", "BiGRU", "FT-Transformer", "simple_avg", "weighted_avg",
        "stacking_xgb_bigru", "stacking_xgb_bigru_foldavg",
        "stacking_all3", "stacking_all3_foldavg", "rerank_top20_union")] or methods
    for name in order:
        a = agg[name]
        lines.append(f"| {name} | {a['acc']['mean']:.4f} ± {a['acc']['std']:.4f} | "
                     f"{a['macro_f1_support']['mean']:.4f} ± {a['macro_f1_support']['std']:.4f} | "
                     f"{a['weighted_f1']['mean']:.4f} | {a['top5']['mean']:.4f} |")
    lines += ["", "## 2. Paired verdict", "",
              f"- Criterion: {verdict['criterion']}",
              f"- ΔAcc vs E4 XGBoost-104: {m_e4:+.2f} pp "
              f"[95% CI {lo_e4:+.2f}, {hi_e4:+.2f}]; per-seed "
              f"{[round(float(x), 2) for x in d_e4]}",
              f"- Δ vs weighted_avg: {m_w:+.2f} pp [95% CI {lo_w:+.2f}, {hi_w:+.2f}]",
              f"- foldavg − bagged: {float(d_bag.mean()):+.2f} pp",
              f"- Point estimate {'meets' if verdict['point_pass'] else 'misses'} +0.5 pp;"
              f"CI lower bound{' > 0' if verdict['ci_lower_pass'] else ' ≤ 0'};"
              f"Formal verdict: {'pass' if verdict['formal_pass'] else 'fail'}",
              "", "## 3. Guardrails (primary protocol)", "",
              f"- Macro-F1(sup) mean {guard['macro_f1_support_mean']:.4f} "
              f"(ref {GUARDRAIL_MACRO_F1_SUP:.4f}); per-seed pass {guard['macro_f1_support_seeds_pass'][0]}/"
              f"{guard['macro_f1_support_seeds_pass'][1]}",
              f"- Top-5 mean {guard['top5_mean']:.4f} (ref {GUARDRAIL_TOP5:.4f}); per-seed pass "
              f"{guard['top5_seeds_pass'][0]}/{guard['top5_seeds_pass'][1]}",
              "", "## 4. Evidence boundary", "",
              "- The degrees of freedom of the 5-seed paired CI is only 4, so intervals are wide; E4 and this experiment share seeds but fold models use only 80% of training data.",
              "- The meta-layer foldavg is the primary convention; bagged serves only as a quantified reference for distribution mismatch.",
              "", "## 5. Artifacts", "",
              "- `E8_seeds_multiseed.csv` (method x seed details)",
              "- `E8_summary_multiseed.json` (machine-readable summary)",
              "- `E8_report_multiseed.md` (this file) "]
    with open(os.path.join(HERE, "E8_report_multiseed.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"multi sub-seed aggregation complete | Δ vs E4 = {m_e4:+.2f} pp [CI {lo_e4:+.2f}, {hi_e4:+.2f}] | "
          f"Formal verdict: {'pass' if verdict['formal_pass'] else 'fail'}", flush=True)


# ==================== Main pipeline ====================
STRATA = None


def main():
    global SUFFIX, LOG, RUNS_CSV, FOLDS_CSV, OOF_NPZ, SUMMARY_JSON, REPORT_MD
    global FIG_PNG, LOG_PATH, STRATA

    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-epochs", type=int, default=0, help="0 = protocol default")
    ap.add_argument("--skip-ftt", action="store_true")
    ap.add_argument("--lr-c", type=float, default=1.0)
    ap.add_argument("--tag", default="", help="extra artifact suffix (replaces the default suffix), for multi sub-seeds")
    ap.add_argument("--seeds", nargs="*", type=int, default=None,
                    help="multi sub-seeds: spawn a subprocess per seed in turn, artifacts suffixed with {tag}seedN")
    ap.add_argument("--analyze-multiseed", action="store_true",
                    help="aggregate completed sub-seed results, compute paired CI and formal verdict")
    args = ap.parse_args()

    if args.analyze_multiseed:
        analyze_multiseed(args.seeds or SEEDS, args.tag or "ms_")
        return
    if args.seeds:
        import subprocess
        tag = args.tag or "ms_"
        for s in args.seeds:
            cmd = [sys.executable, "-B", os.path.abspath(__file__),
                   "--seed", str(s), "--tag", f"{tag}seed{s}"]
            print(f"[multiseed] spawning subprocess: seed={s} -> suffix _{tag}seed{s}", flush=True)
            r = subprocess.run(cmd, cwd=HERE)
            if r.returncode != 0:
                raise RuntimeError(f"sub-seed {s} failed (exit={r.returncode})")
        print(f"[multiseed] all {len(args.seeds)} seeds done; run --analyze-multiseed to aggregate",
              flush=True)
        return

    SUFFIX = "_smoke" if args.smoke else ("_pilot" if args.pilot else "")
    if args.tag:
        SUFFIX = f"_{args.tag}"
    RUNS_CSV = os.path.join(HERE, f"E8_runs{SUFFIX}.csv")
    FOLDS_CSV = os.path.join(HERE, f"E8_folds{SUFFIX}.csv")
    OOF_NPZ = os.path.join(HERE, f"E8_oof{SUFFIX}.npz")
    SUMMARY_JSON = os.path.join(HERE, f"E8_summary{SUFFIX}.json")
    REPORT_MD = os.path.join(HERE, f"E8_report{SUFFIX}.md")
    FIG_PNG = os.path.join(HERE, f"E8_metrics{SUFFIX}.png")
    LOG_PATH = os.path.join(HERE, f"e8_run{SUFFIX}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
                  logging.StreamHandler()])
    LOG = logging.getLogger("E8")

    models = [m for m in MODELS if not (args.skip_ftt and m == "FT-Transformer")]
    smoke = args.smoke
    n_folds = 3 if smoke else args.folds
    max_epochs_cap = (args.max_epochs or (1 if smoke else 0))

    log("=" * 78)
    log(f"E8 probability stacking | mode={'smoke' if smoke else ('pilot' if args.pilot else 'full')} "
        f"| seed={args.seed} | folds={n_folds} | models={models}")
    log(f"device={ec.DEVICE} | torch={torch.__version__}")

    t_start = time.time()
    raw = load_raw()
    log(f"data: train {raw['X_train'].shape} | val {raw['X_val'].shape} | "
        f"test {raw['X_test'].shape} | classes {N_CLASSES}")
    y_train, y_val, y_test = raw["y_train"], raw["y_val"], raw["y_test"]
    STRATA = ec.make_strata(y_train, N_CLASSES)

    params = load_tuned_params()
    if args.skip_ftt:
        params.pop("FT-Transformer", None)
    log("tuned parameters: XGB n_estimators=%s | BiGRU hidden=%s bs=%s | FTT d_token=%s bs=%s" % (
        params["XGBoost-104"]["n_estimators"], params["BiGRU"]["hidden_size"],
        params["BiGRU"]["batch_size"],
        params.get("FT-Transformer", {}).get("d_token", "-"),
        params.get("FT-Transformer", {}).get("batch_size", "-")))

    folds = make_folds(y_train, n_folds, args.seed)
    if smoke:
        folds = folds[:1]
    log(f"folds: {len(folds)} | per-fold held-out ≈ {len(folds[0][1])}")
    if os.path.exists(FOLDS_CSV):
        os.remove(FOLDS_CSV)  # avoid duplicate fold-level rows when rerunning

    n_tr = len(y_train)
    oof = {m: np.zeros((n_tr, N_CLASSES), dtype=np.float32) for m in models}
    fold_id = np.full(n_tr, -1, dtype=np.int16)
    test_bag = {m: [] for m in models}
    val_bag = {m: [] for m in models}
    fold_rows = []
    n_est_cap = 8 if smoke else None
    frac = 0.25 if smoke else 0.10

    for fi, (idx_tr, idx_ho) in enumerate(folds):
        log("-" * 78)
        log(f"[fold {fi}] train={len(idx_tr)} | holdout={len(idx_ho)}")
        idx_fit, idx_in = inner_split(idx_tr, y_train, args.seed, frac=frac)
        log(f"[fold {fi}] inner-fit={len(idx_fit)} | inner-val={len(idx_in)}")

        scaler = StandardScaler().fit(raw["X_train"][idx_tr])
        X_fit_raw, X_in_raw = raw["X_train"][idx_fit], raw["X_train"][idx_in]
        X_ho_raw, X_te_raw = raw["X_train"][idx_ho], raw["X_test"]
        X_va_raw = raw["X_val"]
        X_fit_s = scaler.transform(X_fit_raw).astype(np.float32)
        X_in_s = scaler.transform(X_in_raw).astype(np.float32)
        X_ho_s = scaler.transform(X_ho_raw).astype(np.float32)
        X_te_s = scaler.transform(X_te_raw).astype(np.float32)
        X_va_s = scaler.transform(X_va_raw).astype(np.float32)
        y_fit, y_in = y_train[idx_fit], y_train[idx_in]

        for m in models:
            tag = f"[fold {fi}][{m} seed={args.seed}]"
            t0 = time.time()
            if m == "XGBoost-104":
                out = train_xgb_fold(
                    params[m], X_fit_raw, y_fit, X_in_raw, y_in, X_ho_raw,
                    X_te_raw, X_va_raw, args.seed, tag, n_estimators_cap=n_est_cap)
            else:
                spec = TORCH_SPECS[m]
                spec_ep = spec["max_epochs"]
                max_ep = min(spec_ep, max_epochs_cap) if max_epochs_cap else spec_ep
                pat = min(spec["patience"], max(1, max_ep // 2)) if max_epochs_cap else spec["patience"]
                out = train_torch_fold(
                    spec, params[m], X_fit_s, y_fit, X_in_s, y_in, X_ho_s,
                    X_te_s, X_va_s, args.seed, max_ep, pat, tag)
            # Record held-out metrics
            m_ho = full_metrics(out["ho_proba"], y_train[idx_ho])
            log(f"{tag} done {time.time() - t0:.1f}s | holdout Acc={m_ho['acc']:.4f} "
                f"MacroF1={m_ho['macro_f1']:.4f} | best_epoch={out['best_epoch']}")
            oof[m][idx_ho] = out["ho_proba"]
            test_bag[m].append(out["te_proba"])
            val_bag[m].append(out["va_proba"])
            fold_rows.append(dict(
                fold=fi, model=m, n_train=len(idx_fit), n_inner=len(idx_in),
                n_holdout=len(idx_ho),
                holdout_acc=round(float(m_ho["acc"]), 6),
                holdout_macro_f1=round(float(m_ho["macro_f1"]), 6),
                best_epoch=int(out["best_epoch"]), epochs_run=int(out["epochs_run"]),
                n_params=int(out["n_params"]), duration_s=round(time.time() - t0, 1)))
            append_row(FOLDS_CSV, fold_rows[-1], FOLD_FIELDS)
        fold_id[idx_ho] = fi

    # ---- Aggregation ----
    covered = fold_id >= 0
    n_cover = int(covered.sum())
    if n_cover == 0:
        raise RuntimeError("OOF covers no training samples")
    if not covered.all():
        log(f"[warn] OOF covers only {n_cover}/{len(fold_id)} training samples (smoke/partial folds), "
            f"OOF-side metrics and the meta-learner use covered samples only")
    y_oof = y_train[covered]
    oof_c = {m: oof[m][covered] for m in models}
    test = {m: np.mean(test_bag[m], axis=0).astype(np.float32) for m in models}
    val = {m: np.mean(val_bag[m], axis=0).astype(np.float32) for m in models}
    log("=" * 78)
    log("OOF/test/val probability aggregation complete (val monitoring only)")

    methods = {}
    for m in models:
        methods[m] = {"oof": oof_c[m], "test": test[m], "val": val[m]}
    methods["simple_avg"] = {"oof": simple_average(oof_c, models),
                             "test": simple_average(test, models),
                             "val": simple_average(val, models)}
    w, w_acc = search_weights(oof_c, y_oof, models)
    log(f"weighted average weights (OOF grid, step 0.05): {np.round(w, 2).tolist()} | OOF Acc={w_acc:.4f}")
    methods["weighted_avg"] = {"oof": weighted_average(oof_c, w, models),
                               "test": weighted_average(test, w, models),
                               "val": weighted_average(val, w, models),
                               "weights": [float(x) for x in w]}
    st3 = lr_stack(oof_c, y_oof, test, val, models, C=args.lr_c,
                   test_folds=test_bag, val_folds=val_bag)
    methods["stacking_all3"] = {"oof": st3["oof"], "test": st3["test"],
                                "val": st3["val"],
                                "lr_iter": st3["n_iter"], "lr_fit_s": st3["fit_s"]}
    if st3["test_foldavg"] is not None:
        # Primary convention: per-fold meta prediction then average (inference input distribution matches OOF)
        methods["stacking_all3_foldavg"] = {"oof": st3["oof"],
                                            "test": st3["test_foldavg"],
                                            "val": st3["val_foldavg"]}
    if len(models) >= 3 or (len(models) == 2 and "FT-Transformer" not in models):
        sub = [m for m in ("XGBoost-104", "BiGRU") if m in models]
        if len(sub) == 2:
            st2 = lr_stack(oof_c, y_oof, test, val, sub, C=args.lr_c,
                           test_folds={m: test_bag[m] for m in sub},
                           val_folds={m: val_bag[m] for m in sub})
            methods["stacking_xgb_bigru"] = {
                "oof": st2["oof"], "test": st2["test"], "val": st2["val"],
                "lr_iter": st2["n_iter"], "lr_fit_s": st2["fit_s"]}
            if st2["test_foldavg"] is not None:
                methods["stacking_xgb_bigru_foldavg"] = {
                    "oof": st2["oof"], "test": st2["test_foldavg"],
                    "val": st2["val_foldavg"]}
    methods["rerank_top20_union"] = {
        "oof": rerank_topk(oof_c, models), "test": rerank_topk(test, models),
        "val": rerank_topk(val, models)}

    # ---- Metrics ----
    run_rows = []
    metrics_all = {}
    for name, d in methods.items():
        metrics_all[name] = {}
        for split, y in (("test", y_test), ("oof", y_oof), ("val", y_val)):
            if d[split] is None:
                continue
            m = full_metrics(d[split], y)
            metrics_all[name][split] = m
            run_rows.append(metric_row(name, split, m))
            log(f"[{name} | {split}] Acc={m['acc']:.4f} MacroF1={m['macro_f1']:.4f} "
                f"F1sup={m['macro_f1_support']:.4f} Top5={m['top5']:.4f}")
    write_csv(RUNS_CSV, run_rows, RUN_FIELDS)

    # ---- oracle / criteria ----
    pred_test = {m: test[m].argmax(axis=1) for m in models}
    pred_oof = {m: oof_c[m].argmax(axis=1) for m in models}
    orc_test, wrong_test = oracle_acc(pred_test, y_test, models)
    orc_oof, wrong_oof = oracle_acc(pred_oof, y_oof, models)
    oracle_pairs = {}
    if len(models) >= 2:
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                key = f"{models[i]}+{models[j]}"
                oracle_pairs[key] = {
                    "test_acc": round(oracle_acc(pred_test, y_test, [models[i], models[j]])[0], 6),
                    "oof_acc": round(oracle_acc(pred_oof, y_oof, [models[i], models[j]])[0], 6)}
    log(f"oracle(test) three-model union Acc={orc_test:.4f} | all three wrong={wrong_test} "
        f"({wrong_test / len(y_test) * 100:.2f}%)")
    log(f"oracle(OOF)  Acc={orc_oof:.4f} | all three wrong={wrong_oof} "
        f"({wrong_oof / len(y_oof) * 100:.2f}%)")

    stack_name = ("stacking_all3_foldavg" if "stacking_all3_foldavg" in metrics_all
                  else "stacking_all3")
    stack_acc = metrics_all[stack_name]["test"]["acc"]
    stack_bag_acc = metrics_all["stacking_all3"]["test"]["acc"]
    bag_xgb_acc = metrics_all["XGBoost-104"]["test"]["acc"]
    delta_vs_e4_pp = (stack_acc - E4_REF_ACC["XGBoost-104"]) * 100
    delta_vs_bag_pp = (stack_acc - bag_xgb_acc) * 100
    guard = {
        "macro_f1_support": round(float(metrics_all[stack_name]["test"]["macro_f1_support"]), 6),
        "macro_f1_support_ref": GUARDRAIL_MACRO_F1_SUP,
        "macro_f1_support_pass": bool(
            metrics_all[stack_name]["test"]["macro_f1_support"]
            >= GUARDRAIL_MACRO_F1_SUP - GUARDRAIL_TOL_PP / 100),
        "top5": round(float(metrics_all[stack_name]["test"]["top5"]), 6),
        "top5_ref": GUARDRAIL_TOP5,
        "top5_pass": bool(metrics_all[stack_name]["test"]["top5"] >= GUARDRAIL_TOP5),
    }
    verdict = {
        "criterion": "delta Acc >= +0.5 pp vs E4 XGBoost-104 mean 0.8333, and the 5-seed paired 95% CI lower bound > 0",
        "pilot_status": "Single-seed point estimate (no formal verdict; CI and multi-sub-seed repetition await the full experiment)",
        "primary_stack_method": stack_name,
        "primary_inference": "Per-fold meta prediction then average (inference input distribution matches OOF)",
        "stacking_test_acc": round(float(stack_acc), 6),
        "stacking_bagged_test_acc": round(float(stack_bag_acc), 6),
        "delta_foldavg_minus_bagged_pp": round(float(stack_acc - stack_bag_acc) * 100, 3),
        "xgb_bagged_test_acc": round(float(bag_xgb_acc), 6),
        "delta_vs_e4_ref_pp": round(float(delta_vs_e4_pp), 3),
        "delta_vs_bagged_xgb_pp": round(float(delta_vs_bag_pp), 3),
        "point_pass": bool(delta_vs_e4_pp >= DELTA_PP_REQUIRED),
        "guardrail": guard,
    }

    # ---- Persist to disk ----
    np.savez_compressed(
        OOF_NPZ,
        fold_id=fold_id, y_train=y_train, y_val=y_val, y_test=y_test,
        **{f"oof_{SLUG[m]}": oof[m].astype(np.float16) for m in models},
        **{f"test_{SLUG[m]}": test[m].astype(np.float16) for m in models},
        **{f"fold_test_{SLUG[m]}": np.stack(test_bag[m]).astype(np.float16) for m in models},
        **{f"fold_val_{SLUG[m]}": np.stack(val_bag[m]).astype(np.float16) for m in models},
        **{f"stack_oof_{k}": methods[k]["oof"].astype(np.float16)
           for k in methods if k.startswith("stacking")},
        **{f"stack_test_{k}": methods[k]["test"].astype(np.float16)
           for k in methods if k.startswith("stacking")},
        **{f"stack_val_{k}": methods[k]["val"].astype(np.float16)
           for k in methods
           if k.startswith("stacking") and methods[k]["val"] is not None},
    )

    summary = {
        "experiment": "E8", "mode": "smoke" if smoke else ("pilot" if args.pilot else "full"),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": int(args.seed), "n_folds": int(len(folds)),
        "models": models, "lr_C": float(args.lr_c),
        "protocol": {
            "data": "ZEOSYN_104_struct_cache.npz (104 features, fillna(0)), same source as E4",
            "outer_cv": "StratifiedKFold(shuffle=True, random_state=seed), held-out folds only produce OOF",
            "inner_split": f"Within-fold {int(frac * 100)}% stratified inner validation set, used only for early stopping / tree count",
            "scaler": "StandardScaler fit within folds only (deep models); XGBoost uses raw features",
            "test_probs": ("Per-fold models emit test/val probabilities (saved per fold in the npz as fold_test_*/fold_val_*), "
                           "The 5-fold arithmetic mean gives bagged probabilities; the meta layer reports two inference modes: "
                           "'infer on bagged probabilities' vs 'per-fold meta prediction then average (primary)'"),
            "meta_inference": ("Primary foldavg: each fold's base-model probabilities pass the meta-learner separately, then the 5-fold meta outputs are averaged, "
                               "so the inference input (single-fold distribution) matches the training input (OOF single-fold distribution); "
                               "bagged serves as reference only "),
            "meta": "LogisticRegression(multinomial,lbfgs,C=%s,max_iter=2000) fit on OOF" % args.lr_c,
        },
        "params": {k: {kk: (float(vv) if isinstance(vv, (int, float)) else vv)
                       for kk, vv in v.items()} for k, v in params.items()},
        "reference_E4_acc": {k: {"mean": v, "std": E4_REF_STD[k]} for k, v in E4_REF_ACC.items()},
        "weights": methods["weighted_avg"]["weights"],
        "metrics": {name: {split: {k: (None if isinstance(v, float) and math.isnan(v) else v)
                                   for k, v in mm.items()}
                           for split, mm in metrics_all[name].items()}
                    for name in metrics_all},
        "oracle": {
            "hist": ORACLE_HIST,
            "test": {"all3_acc": round(float(orc_test), 6),
                     "all3_wrong_n": int(wrong_test), "n_test": int(len(y_test))},
            "oof": {"all3_acc": round(float(orc_oof), 6),
                    "all3_wrong_n": int(wrong_oof), "n_train": int(len(y_oof))},
            "pairs": oracle_pairs,
        },
        "verdict": verdict,
        "timings": {"total_s": round(time.time() - t_start, 1),
                    "folds": fold_rows},
        "artifacts": {"runs_csv": RUNS_CSV, "folds_csv": FOLDS_CSV, "oof_npz": OOF_NPZ,
                      "figure": FIG_PNG, "log": LOG_PATH},
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    make_figure(metrics_all, y_test)
    write_report(summary)
    log(f"done | total time {summary['timings']['total_s']}s | artifacts: "
        f"{os.path.basename(RUNS_CSV)}, {os.path.basename(SUMMARY_JSON)}, "
        f"{os.path.basename(REPORT_MD)}, {os.path.basename(FIG_PNG)}")


def make_figure(metrics_all, y_test):
    names = list(metrics_all.keys())
    acc = [metrics_all[n]["test"]["acc"] * 100 for n in names]
    f1s = [metrics_all[n]["test"]["macro_f1_support"] * 100 for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(9, len(names) * 1.15), 4.6))
    wbar = 0.38
    b1 = ax.bar(x - wbar / 2, acc, wbar, label="Test Acc (%)", color="#4c72b0")
    b2 = ax.bar(x + wbar / 2, f1s, wbar, label="Macro-F1 (supported, %)", color="#dd8452")
    for bars in (b1, b2):
        for r in bars:
            ax.annotate(f"{r.get_height():.2f}", (r.get_x() + r.get_width() / 2, r.get_height()),
                        ha="center", va="bottom", fontsize=8)
    ax.axhline(E4_REF_ACC["XGBoost-104"] * 100, ls="--", lw=1, color="#55a868",
               label=f"E4 XGBoost ref {E4_REF_ACC['XGBoost-104'] * 100:.2f}%")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("percent")
    ax.set_title("E8 stacking: test metrics (single seed pilot)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_PNG, dpi=150)
    plt.close(fig)


def write_report(summary):
    v = summary["verdict"]
    g = v["guardrail"]
    modes = {"smoke": "smoke check", "pilot": "single-seed pilot", "full": "full"}
    lines = []
    lines.append(f"# E8 probability stacking ({summary['mode']}) report\n")
    lines.append(f"- Mode: {modes[summary['mode']]} | seed={summary['seed']} | "
                 f"folds={summary['n_folds']} | models: {', '.join(summary['models'])}")
    lines.append(f"- Date: {summary['date']}")
    lines.append("")
    lines.append("## 1. Protocol\n")
    for k, val in summary["protocol"].items():
        lines.append(f"- **{k}**: {val}")
    lines.append("")
    lines.append("## 2. Test set results\n")
    lines.append("| Method | Acc | Macro-F1(sup) | Weighted-F1 | Top-5 |")
    lines.append("|---|---|---|---|---|")
    order = [n for n in summary["metrics"] if n in (
        "XGBoost-104", "BiGRU", "FT-Transformer", "simple_avg", "weighted_avg",
        "stacking_xgb_bigru", "stacking_xgb_bigru_foldavg",
        "stacking_all3", "stacking_all3_foldavg", "rerank_top20_union")]
    for name in order:
        if name not in summary["metrics"] or "test" not in summary["metrics"][name]:
            continue
        m = summary["metrics"][name]["test"]
        lines.append(f"| {name} | {m['acc']:.4f} | {m['macro_f1_support']:.4f} | "
                     f"{m['weighted_f1']:.4f} | {m['top5']:.4f} |")
    lines.append("")
    lines.append("## 3. Criteria and guardrails\n")
    lines.append(f"- Criterion: {v['criterion']}")
    lines.append(f"- Status: {v['pilot_status']}")
    lines.append(f"- stacking test Acc (primary metric {v['primary_stack_method']}) = "
                 f"{v['stacking_test_acc']:.4f}; bagged inference reference = "
                 f"{v['stacking_bagged_test_acc']:.4f}; "
                 f"5-fold bagged XGBoost = {v['xgb_bagged_test_acc']:.4f}")
    lines.append(f"- foldavg − bagged (meta-layer inference metric gap) = "
                 f"{v['delta_foldavg_minus_bagged_pp']:+.2f} pp")
    lines.append(f"- ΔAcc vs E4 XGBoost mean = {v['delta_vs_e4_ref_pp']:+.2f} pp "
                 f"(point estimate {'meets' if v['point_pass'] else 'misses'} the +0.5 pp line); "
                 f"Δ vs bagged XGBoost under this protocol = {v['delta_vs_bagged_xgb_pp']:+.2f} pp")
    lines.append(f"- guardrail Macro-F1(sup) = {g['macro_f1_support']:.4f} "
                 f"(ref {g['macro_f1_support_ref']:.4f}, {'pass' if g['macro_f1_support_pass'] else 'fail'}); "
                 f"Top-5 = {g['top5']:.4f} (ref {g['top5_ref']:.4f}, "
                 f"{'pass' if g['top5_pass'] else 'fail'})")
    lines.append("")
    lines.append("## 4. Oracle upper bound and complementarity\n")
    o = summary["oracle"]
    lines.append(f"- three-model union oracle (test) = {o['test']['all3_acc']:.4f}; "
                 f"all three wrong {o['test']['all3_wrong_n']}/{o['test']['n_test']} "
                 f"({o['test']['all3_wrong_n'] / o['test']['n_test'] * 100:.2f}%)")
    lines.append(f"- three-model union oracle (OOF) = {o['oof']['all3_acc']:.4f}; "
                 f"all three wrong {o['oof']['all3_wrong_n']}/{o['oof']['n_train']} "
                 f"({o['oof']['all3_wrong_n'] / o['oof']['n_train'] * 100:.2f}%)")
    for k, d in o["pairs"].items():
        lines.append(f"- oracle {k}: test {d['test_acc']:.4f} / OOF {d['oof_acc']:.4f}")
    lines.append(f"- E4 historical upper-bound reference: {json.dumps(o['hist'], ensure_ascii=False)}")
    lines.append("")
    lines.append("## 5. Evidence boundary\n")
    lines.append("- single seed: no confidence interval, cannot declare the \"dividend holds\" from this; formal conclusion needs a 5-seed paired CI.")
    lines.append("- fold models train on 80% of data only, so single-model bagged Acc is lower than E4's full-training-set result, "
                 "The two conventions must be reported separately. ")
    lines.append("- the meta-learner fits on OOF, its OOF metrics are in-fit samples, reference only; criteria always use test.")
    lines.append("- two meta-layer inference metrics side by side: foldavg (primary, per-fold meta predictions averaged, consistent with the OOF training distribution) "
                 "vs bagged (inferring on the 5-fold averaged probabilities, higher entropy). The difference itself quantifies the distribution mismatch. ")
    lines.append("- val probabilities are bagged from per-fold models, monitoring only, involved in no selection.")
    lines.append("")
    lines.append("## 6. Artifacts\n")
    for k, val in summary["artifacts"].items():
        lines.append(f"- `{os.path.basename(val)}` ({k})")
    lines.append("")
    lines.append("## 7. Per-fold timing\n")
    lines.append("| fold | model | fit n | inner-val n | holdout n | Acc | Macro-F1 | best_epoch | s |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in summary["timings"]["folds"]:
        lines.append(f"| {r['fold']} | {r['model']} | {r['n_train']} | {r['n_inner']} | "
                     f"{r['n_holdout']} | {r['holdout_acc']:.4f} | "
                     f"{r['holdout_macro_f1']:.4f} | {r['best_epoch']} | {r['duration_s']} |")
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    import traceback
    rc = 0
    try:
        main()
    except BaseException:
        traceback.print_exc()
        rc = 1
    # Under WDDM, CUDA context teardown crashes occasionally; exit directly to avoid it (bypass teardown on both success and failure)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
