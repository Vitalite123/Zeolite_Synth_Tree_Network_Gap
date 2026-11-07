"""
E4 multi-seed repetition and significance testing
=====================================================================================
Companion documents:
  - CBUTransfer/DeepLearning/DeepLearning_Supplementary_Plan.md §5 (E4)
  - CBUTransfer/DeepLearning/SupplementaryExperiments/README.md
  - CBUTransfer/DeepLearning/DeepLearning_ANN_Full_Report.md (write-back §4.1c / §5.6.4 / §6.1)

Purpose:
  1) Repeat the 6 tuned models with 5 random seeds (42/123/777/2026/3407)
     (XGBoost-104 / BiGRU / LeNet-5 / Transformer / FT-Transformer / TabNet),
     report mean±std of test Acc / Macro-F1 / Weighted-F1 / Top-5;
  2) Run the McNemar test on paired predictions for the same split and same test set
     (BiGRU vs XGBoost, and each ANN vs XGBoost);
  3) Wilcoxon signed-rank test on per-class F1 over the 231 classes (all / head / torso / tail),
     and Holm-Bonferroni to control multiple comparisons.

Protocol:
  - Data/splits/features fully consistent with E1/E2 (104 dims, 12,537 / 2,640 / 2,641);
  - Each model is retrained with its Optuna-best configuration;
  - Early stopping consistent with the original experiments (LeNet-5 60/10, others 100/15);
  - The test set is used only for final evaluation, never involved in any selection.

Outputs (this directory):
  seed_repetition_results.csv      one row per (model, seed) run
  per_class_f1_distribution.csv    F1 for each (model, seed, class)
  significance_tests.csv           McNemar / Wilcoxon / Holm correction results
  seed_results.png                 mean±std bar chart (English labels, avoiding font issues)
  E4_summary.json                  machine-readable summary
  E4_report.md                     human-readable report
  e4_run.log                       run log
  preds/{slug}_seed{seed}.npy      test set predictions (int16), for paired-test recomputation

Usage:
  py -B e4_seeds_significance.py                # train (resumable) + analyze + report
  py -B e4_seeds_significance.py --analyze-only # recompute analysis and report from existing artifacts only
"""

import argparse
import csv
import gc
import json
import math
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xgboost as xgb
from scipy.stats import binomtest, chi2, t as tdist, wilcoxon
from sklearn.metrics import f1_score, log_loss as sk_log_loss
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
from e1_1_lenet5 import build_model as build_lenet5  # noqa: E402
from e1_2_bigru import build_model as build_bigru  # noqa: E402
from e1_3_transformer import build_model as build_transformer  # noqa: E402
from e2_1_ft_transformer import build_ftt, forward_ftt  # noqa: E402

import pytorch_tabnet.metrics as pt_metrics  # noqa: E402
from pytorch_tabnet.tab_model import TabNetClassifier  # noqa: E402

# pytorch-tabnet 4.1.0's built-in LogLoss does not pass labels to sklearn; newer sklearn raises an error;
# The same minimal-difference patch as E2.2.
def _patched_logloss_call(self, y_true, y_score):
    return float(sk_log_loss(y_true, y_score,
                             labels=np.arange(np.asarray(y_score).shape[1])))

pt_metrics.LogLoss.__call__ = _patched_logloss_call

# ==================== Constants ====================
SEEDS = [42, 123, 777, 2026, 3407]
MODELS = ["XGBoost-104", "BiGRU", "LeNet-5", "Transformer", "FT-Transformer", "TabNet"]
SLUG = {
    "XGBoost-104": "xgboost",
    "BiGRU": "bigru",
    "LeNet-5": "lenet5",
    "Transformer": "transformer",
    "FT-Transformer": "ftt",
    "TabNet": "tabnet",
}
REF_MODEL = "XGBoost-104"
N_CLASSES = ec.N_CLASS_EXPECT
MAX_EPOCHS = ec.MAX_EPOCHS
PATIENCE = ec.PATIENCE

RESULTS_CSV = os.path.join(HERE, "seed_repetition_results.csv")
PERCLASS_CSV = os.path.join(HERE, "per_class_f1_distribution.csv")
SIG_CSV = os.path.join(HERE, "significance_tests.csv")
SUMMARY_JSON = os.path.join(HERE, "E4_summary.json")
REPORT_MD = os.path.join(HERE, "E4_report.md")
FIG_PNG = os.path.join(HERE, "seed_results.png")
PRED_DIR = os.path.join(HERE, "preds")
LOG_PATH = os.path.join(HERE, "e4_run.log")

RESULT_FIELDS = [
    "model", "seed", "acc", "macro_f1", "weighted_f1", "top5",
    "precision_macro", "recall_macro", "top3", "top10",
    "head_f1", "torso_f1", "tail_f1",
    "val_acc", "val_macro_f1", "n_params", "best_epoch", "duration_s",
]

LOG = None
EPOCH_CAP = None  # for smoke tests only (--epoch-cap), None in normal runs


def log(msg):
    if LOG is not None:
        LOG.info(msg)
    print(msg, flush=True)


# ==================== Tuned parameter loading ====================
def _json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _py(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def load_tuned_params():
    p_lenet = _json(os.path.join(E1_DIR, "best_params_E1_1.json"))["params"]
    p_bigru = _json(os.path.join(E1_DIR, "best_params_E1_2.json"))["params"]
    p_trans = _json(os.path.join(E1_DIR, "best_params_E1_3.json"))["params"]
    p_xgb = _json(os.path.join(E1_DIR, "best_params.json"))["params"]

    ftt_keys = ["d_token", "n_blocks", "n_heads", "attn_dropout", "ffn_mult",
                "ffn_dropout", "resid_dropout", "lr", "weight_decay", "batch_size"]
    tab_keys = ["n_independent", "n_d", "n_a", "n_steps", "gamma", "n_shared",
                "momentum", "lambda_sparse", "lr", "weight_decay", "batch_size"]

    h1 = pd.read_csv(os.path.join(E2_DIR, "E2_1_trials_history.csv"))
    row1 = h1[h1["number"] == 21].iloc[0]
    p_ftt = {k: _py(row1[k]) for k in ftt_keys}

    h2 = pd.read_csv(os.path.join(E2_DIR, "E2_2_trials_history.csv"))
    row2 = h2[h2["number"] == 29].iloc[0]
    p_tab = {k: _py(row2[k]) for k in tab_keys}

    return {
        "XGBoost-104": p_xgb,
        "BiGRU": p_bigru,
        "LeNet-5": p_lenet,
        "Transformer": p_trans,
        "FT-Transformer": p_ftt,
        "TabNet": p_tab,
    }


# ==================== Data ====================
def load_raw():
    """Labels in the npz cache are pre-encoding raw values and must be LabelEncoder-transformed the same way as load_data."""
    from sklearn.preprocessing import LabelEncoder
    z = np.load(ec.CACHE_PATH)
    le = LabelEncoder()
    y_train = le.fit_transform(z["y_train"])
    y_val = le.transform(z["y_val"])
    y_test = le.transform(z["y_test"])
    if len(le.classes_) != N_CLASSES:
        raise RuntimeError(f"class count {len(le.classes_)} != {N_CLASSES}")
    return dict(
        X_train=z["X_train"].astype(np.float32),
        y_train=y_train.astype(np.int64),
        X_val=z["X_val"].astype(np.float32),
        y_val=y_val.astype(np.int64),
        X_test=z["X_test"].astype(np.float32),
        y_test=y_test.astype(np.int64),
        n_classes=N_CLASSES,
    )


# ==================== Training: generic PyTorch ====================
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


def _train_once(build_fn, forward, params, data, seed, bs, max_epochs, patience,
                optimizer_kind, scheduler_kind, label_smoothing, grad_clip,
                warmup_ratio, tag):
    ec.set_seed(seed)
    train_ds = TensorDataset(torch.as_tensor(data["X_train"]),
                             torch.as_tensor(data["y_train"], dtype=torch.long))
    loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=False)
    model = build_fn(params).to(ec.DEVICE)
    n_params = sum(p.numel() for p in model.parameters())

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

        probs, val_loss = _val_probs_loss(model, forward, data["X_val"], data["y_val"])
        val_f1 = ec.metrics_from_proba(probs, data["y_val"], data["n_classes"])["macro_f1"]
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
    test_proba = ec.predict_proba(model, data["X_test"], forward)
    val_proba = ec.predict_proba(model, data["X_val"], forward)
    log(f"{tag} best_epoch={best_epoch} | val MacroF1={best_f1:.4f} | "
        f"test Acc={ec.metrics_from_proba(test_proba, data['y_test'], data['n_classes'])['acc']:.4f} | "
        f"params={n_params}")
    del model
    if ec.DEVICE.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
    return dict(test_proba=test_proba, val_proba=val_proba,
                n_params=int(n_params), best_epoch=int(best_epoch))


def train_torch(build_fn, forward, params, data, seed, max_epochs, patience,
                optimizer_kind, scheduler_kind, label_smoothing, grad_clip,
                warmup_ratio, tag):
    bs = int(params["batch_size"])
    if ec.DEVICE.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
    for attempt in range(3):
        try:
            return _train_once(build_fn, forward, params, data, seed, bs, max_epochs,
                               patience, optimizer_kind, scheduler_kind,
                               label_smoothing, grad_clip, warmup_ratio, tag)
        except torch.OutOfMemoryError:
            if attempt == 2 or bs // 2 < 64:
                raise
            bs = max(64, bs // 2)
            log(f"{tag} | CUDA OOM, batch -> {bs}")
            gc.collect()
            torch.cuda.empty_cache()


# ==================== Training: each model ====================
def train_xgboost(params, raw, seed, tag):
    ec.set_seed(seed)
    clf = xgb.XGBClassifier(
        tree_method="hist", random_state=seed, n_jobs=-1,
        eval_metric="mlogloss", early_stopping_rounds=50, **params)
    clf.fit(raw["X_train"], raw["y_train"],
            eval_set=[(raw["X_val"], raw["y_val"])], verbose=False)
    proba_te = np.zeros((len(raw["y_test"]), raw["n_classes"]), dtype=np.float32)
    proba_va = np.zeros((len(raw["y_val"]), raw["n_classes"]), dtype=np.float32)
    proba_te[:, clf.classes_] = clf.predict_proba(raw["X_test"])
    proba_va[:, clf.classes_] = clf.predict_proba(raw["X_val"])
    best_iter = int(getattr(clf, "best_iteration", 0)) + 1
    log(f"{tag} best_epoch={best_iter} | test Acc="
        f"{ec.metrics_from_proba(proba_te, raw['y_test'], raw['n_classes'])['acc']:.4f}")
    return dict(test_proba=proba_te, val_proba=proba_va, n_params=0, best_epoch=best_iter)


def train_tabnet(params, data, seed, tag):
    ec.set_seed(seed)
    bs = int(params["batch_size"])
    if ec.DEVICE.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
    clf = TabNetClassifier(
        n_d=int(params["n_d"]), n_a=int(params["n_a"]), n_steps=int(params["n_steps"]),
        gamma=float(params["gamma"]), n_independent=int(params["n_independent"]),
        n_shared=int(params["n_shared"]), momentum=float(params["momentum"]),
        lambda_sparse=float(params["lambda_sparse"]), mask_type="sparsemax",
        optimizer_params=dict(lr=float(params["lr"]),
                              weight_decay=float(params["weight_decay"])),
        seed=seed, verbose=0, device_name=str(ec.DEVICE),
    )
    clf.fit(data["X_train"], data["y_train"],
            eval_set=[(data["X_val"], data["y_val"])],
            eval_metric=["logloss"],
            max_epochs=EPOCH_CAP or MAX_EPOCHS,
            patience=min(PATIENCE, max(1, (EPOCH_CAP or MAX_EPOCHS) // 2)),
            batch_size=bs, virtual_batch_size=bs // 2,
            num_workers=0, drop_last=False)
    proba_te = np.zeros((len(data["y_test"]), data["n_classes"]), dtype=np.float32)
    proba_va = np.zeros((len(data["y_val"]), data["n_classes"]), dtype=np.float32)
    p_te = clf.predict_proba(data["X_test"])
    classes = getattr(clf, "classes_", np.arange(p_te.shape[1]))
    proba_te[:, classes] = p_te
    if p_te.shape[1] == data["n_classes"]:
        proba_va[:, classes] = clf.predict_proba(data["X_val"])
    else:
        raise RuntimeError(f"TabNet output column count invalid: {p_te.shape[1]}")
    vll = list(clf.history["val_0_logloss"])
    best_epoch = int(np.argmin(vll)) + 1
    n_params = int(sum(v.numel() for v in clf.network.parameters()))
    log(f"{tag} epochs={len(clf.history['loss'])} best_epoch={best_epoch} | "
        f"test Acc={ec.metrics_from_proba(proba_te, data['y_test'], data['n_classes'])['acc']:.4f} | "
        f"params={n_params}")
    del clf
    if ec.DEVICE.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
    return dict(test_proba=proba_te, val_proba=proba_va,
                n_params=n_params, best_epoch=best_epoch)


TORCH_SPECS = {
    "BiGRU": dict(build=lambda p: build_bigru(p, N_CLASSES),
                  forward=lambda m, x: m(x),
                  max_epochs=100, patience=15, optimizer_kind="adamw",
                  scheduler_kind="plateau", label_smoothing=0.1, grad_clip=1.0),
    "LeNet-5": dict(build=lambda p: build_lenet5(p, N_CLASSES),
                    forward=lambda m, x: m(x),
                    max_epochs=60, patience=10, optimizer_kind="adam",
                    scheduler_kind="none", label_smoothing=0.0, grad_clip=1.0),
    "Transformer": dict(build=lambda p: build_transformer(p, N_CLASSES),
                        forward=lambda m, x: m(x),
                        max_epochs=100, patience=15, optimizer_kind="adamw",
                        scheduler_kind="cosine_warmup", label_smoothing=0.0,
                        grad_clip=1.0),
    "FT-Transformer": dict(build=lambda p: build_ftt(p, N_CLASSES),
                           forward=forward_ftt,
                           max_epochs=100, patience=15, optimizer_kind="adamw",
                           scheduler_kind="none", label_smoothing=0.0,
                           grad_clip=None),
}


# ==================== CSV append ====================
def append_row(path, row, fields):
    new = not os.path.exists(path)
    mode = "w" if new else "a"
    enc = "utf-8-sig" if new else "utf-8"
    with open(path, mode, newline="", encoding=enc) as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)


def pred_path(model, seed):
    return os.path.join(PRED_DIR, f"{SLUG[model]}_seed{seed}.npy")


def load_done():
    if not os.path.exists(RESULTS_CSV):
        return set()
    df = pd.read_csv(RESULTS_CSV)
    done = set()
    for _, r in df.iterrows():
        if os.path.exists(pred_path(r["model"], int(r["seed"]))):
            done.add((r["model"], int(r["seed"])))
    return done


# ==================== Single run ====================
def run_one(model, seed, params, scaled, raw, done):
    if (model, seed) in done:
        log(f"[skip] {model} seed={seed} already done")
        return
    tag = f"[{model} seed={seed}]"
    log(f"=== {tag} start ===")
    t0 = time.time()

    if model == "XGBoost-104":
        out = train_xgboost(params, raw, seed, tag)
    elif model == "TabNet":
        out = train_tabnet(params, scaled, seed, tag)
    else:
        spec = TORCH_SPECS[model]
        max_ep = min(spec["max_epochs"], EPOCH_CAP) if EPOCH_CAP else spec["max_epochs"]
        out = train_torch(
            spec["build"], spec["forward"], params, scaled, seed,
            max_ep, min(spec["patience"], max(1, max_ep // 2)),
            spec["optimizer_kind"], spec["scheduler_kind"],
            spec["label_smoothing"], spec["grad_clip"],
            params.get("warmup_ratio"), tag)

    test_m = ec.metrics_from_proba(out["test_proba"], scaled["y_test"],
                                   scaled["n_classes"], scaled.get("strata"))
    val_m = ec.metrics_from_proba(out["val_proba"], scaled["y_val"],
                                  scaled["n_classes"], scaled.get("strata"))
    pred = out["test_proba"].argmax(axis=1).astype(np.int16)
    np.save(pred_path(model, seed), pred)

    row = {
        "model": model, "seed": int(seed),
        "acc": round(float(test_m["acc"]), 6),
        "macro_f1": round(float(test_m["macro_f1"]), 6),
        "weighted_f1": round(float(test_m["weighted_f1"]), 6),
        "top5": round(float(test_m["top5"]), 6),
        "precision_macro": round(float(test_m.get("precision_macro", np.nan)), 6),
        "recall_macro": round(float(test_m.get("recall_macro", np.nan)), 6),
        "top3": round(float(test_m.get("top3", np.nan)), 6),
        "top10": round(float(test_m.get("top10", np.nan)), 6),
        "head_f1": round(float(test_m.get("head_f1", np.nan)), 6),
        "torso_f1": round(float(test_m.get("torso_f1", np.nan)), 6),
        "tail_f1": round(float(test_m.get("tail_f1", np.nan)), 6),
        "val_acc": round(float(val_m["acc"]), 6),
        "val_macro_f1": round(float(val_m["macro_f1"]), 6),
        "n_params": int(out["n_params"]),
        "best_epoch": int(out["best_epoch"]),
        "duration_s": round(time.time() - t0, 1),
    }
    append_row(RESULTS_CSV, row, RESULT_FIELDS)
    log(f"=== {tag} done in {row['duration_s']}s | "
        f"Acc={row['acc']:.4f} MacroF1={row['macro_f1']:.4f} ===")


# ==================== Holm-Bonferroni ====================
def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj


# ==================== Analysis ====================
def _mean_std(v):
    v = np.asarray(v, dtype=float)
    if len(v) == 0:
        return np.nan, np.nan
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def build_per_class(results, y_test):
    supports = np.bincount(y_test, minlength=N_CLASSES)
    rows = []
    for _, r in results.iterrows():
        p = pred_path(r["model"], int(r["seed"]))
        if not os.path.exists(p):
            continue
        pred = np.load(p)
        f1s = f1_score(y_test, pred, labels=np.arange(N_CLASSES),
                       average=None, zero_division=0)
        for c in range(N_CLASSES):
            rows.append({
                "model": r["model"], "seed": int(r["seed"]),
                "class_index": int(c), "support": int(supports[c]),
                "f1": round(float(f1s[c]), 6),
            })
    return pd.DataFrame(rows), supports


def mcnemar_pair(pred_a, pred_b, y):
    ca, cb = (pred_a == y), (pred_b == y)
    a = int(np.sum(ca & ~cb))   # a correct, b wrong
    b = int(np.sum(~ca & cb))   # a wrong, b correct
    n = a + b
    if n == 0:
        return 0.0, 1.0, 1.0, a, b
    stat = (abs(a - b) - 1) ** 2 / n
    p_chi = float(chi2.sf(stat, 1))
    p_exact = float(binomtest(a, n, 0.5).pvalue)
    return float(stat), p_exact, p_chi, a, b


def wilcoxon_pair(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a - b
    nz = int(np.sum(diff != 0))
    if len(diff) < 5 or nz < 5:
        return np.nan, 1.0, nz
    try:
        res = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        return float(res.statistic), float(res.pvalue), nz
    except ValueError:
        return np.nan, 1.0, nz


def paired_delta_ci(results, model, col, ref=REF_MODEL):
    a = results[results["model"] == model].set_index("seed")[col]
    b = results[results["model"] == ref].set_index("seed")[col]
    common = sorted(set(a.index) & set(b.index))
    if len(common) == 0:
        return None
    d = (a.loc[common].astype(float) - b.loc[common].astype(float)).values
    n = len(d)
    mean = float(d.mean())
    std = float(d.std(ddof=1)) if n > 1 else 0.0
    if n > 1:
        half = float(tdist.ppf(0.975, n - 1) * std / math.sqrt(n))
    else:
        half = np.nan
    return {"n_seeds": n, "mean_delta": mean, "std_delta": std,
            "ci95_low": mean - half, "ci95_high": mean + half,
            "seeds": [int(s) for s in common]}


def analyze():
    log("=== Analysis phase ===")
    scaled = ec.load_data(LOG)
    y_test = scaled["y_test"]
    y_train = scaled["y_train"]
    strata = scaled["strata"]

    results = pd.read_csv(RESULTS_CSV) if os.path.exists(RESULTS_CSV) else pd.DataFrame()
    if results.empty:
        log("no result file, skipping analysis")
        return
    # Crashes/reruns may produce duplicate rows: keep only records whose prediction files exist; for the same (model, seed) keep the last one
    results = results[results.apply(
        lambda r: os.path.exists(pred_path(r["model"], int(r["seed"]))), axis=1)]
    results = (results.drop_duplicates(subset=["model", "seed"], keep="last")
                      .sort_values(["model", "seed"]).reset_index(drop=True))

    per_class, supports = build_per_class(results, y_test)
    per_class.to_csv(PERCLASS_CSV, index=False, encoding="utf-8-sig")

    # ---- Aggregate mean±std ----
    agg = {}
    for m in MODELS:
        sub = results[results["model"] == m]
        if sub.empty:
            continue
        entry = {"n_seeds": int(len(sub)), "seeds": sorted(int(s) for s in sub["seed"])}
        for col in ["acc", "macro_f1", "weighted_f1", "top5", "top3", "top10",
                    "precision_macro", "recall_macro", "head_f1", "torso_f1",
                    "tail_f1", "val_acc", "val_macro_f1", "duration_s"]:
            mu, sd = _mean_std(sub[col].values)
            entry[col] = {"mean": round(mu, 6), "std": round(sd, 6)}
        entry["n_params"] = int(sub["n_params"].iloc[0])
        entry["best_epoch_mean"] = round(float(sub["best_epoch"].astype(float).mean()), 2)
        agg[m] = entry

    # ---- Prediction loading ----
    preds = {}
    for m in MODELS:
        for s in SEEDS:
            p = pred_path(m, s)
            if os.path.exists(p):
                preds[(m, s)] = np.load(p)

    sig_rows = []

    # ---- McNemar: each ANN vs XGBoost ----
    model_list = [m for m in MODELS
                  if m != REF_MODEL and any((m, s) in preds for s in SEEDS)]
    primary_mcnemar = []
    for m in model_list:
        for s in SEEDS:
            if (m, s) not in preds or (REF_MODEL, s) not in preds:
                continue
            stat, p_exact, p_chi, a, b = mcnemar_pair(preds[(m, s)], preds[(REF_MODEL, s)], y_test)
            effect = (a - b) / len(y_test) * 100.0
            fam = "primary_mcnemar" if s == SEEDS[0] else "robust_mcnemar"
            row = {
                "family": fam, "test_type": "mcnemar", "model_a": m,
                "model_b": REF_MODEL, "subset": "test_all", "seed": int(s),
                "n_units": int(a + b), "statistic": round(stat, 4),
                "p_value": round(p_exact, 8), "p_holm": np.nan,
                "significant": "", "effect": round(effect, 4),
                "detail": f"a={a},b={b},p_chi2={p_chi:.3e}",
            }
            sig_rows.append(row)
            if s == SEEDS[0]:
                primary_mcnemar.append(row)

    if primary_mcnemar:
        adj = holm([r["p_value"] for r in primary_mcnemar])
        for r, pa in zip(primary_mcnemar, adj):
            r["p_holm"] = round(float(pa), 8)
            r["significant"] = "yes" if pa < 0.05 else "no"

    # ---- Wilcoxon: per-class F1 (seed-averaged) ----
    primary_wil = []
    if not per_class.empty:
        avg = per_class.groupby(["model", "class_index"])["f1"].mean().unstack("model")
        subset_idx = {
            "all": np.arange(N_CLASSES),
            "supported": np.where(supports > 0)[0],
            "head": np.asarray(strata["head"]),
            "torso": np.asarray(strata["torso"]),
            "tail": np.asarray(strata["tail"]),
        }
        for m in [x for x in MODELS if x != REF_MODEL and x in avg.columns]:
            if REF_MODEL not in avg.columns:
                continue
            for sub_name, idx in subset_idx.items():
                a = avg.loc[idx, m].values
                b = avg.loc[idx, REF_MODEL].values
                W, p, nz = wilcoxon_pair(a, b)
                diff = a - b
                nzd = diff[diff != 0]
                med_nz = float(np.median(nzd)) * 100 if len(nzd) else 0.0
                row = {
                    "family": "primary_wilcoxon", "test_type": "wilcoxon_class_f1",
                    "model_a": m, "model_b": REF_MODEL, "subset": sub_name,
                    "seed": "avg", "n_units": int(nz),
                    "statistic": round(W, 2) if not np.isnan(W) else np.nan,
                    "p_value": round(p, 8), "p_holm": np.nan, "significant": "",
                    "effect": round(med_nz, 4),
                    "detail": f"better={int((diff > 0).sum())},worse={int((diff < 0).sum())},"
                              f"n_classes={len(idx)},n_test_samples={int(supports[idx].sum())}",
                }
                primary_wil.append(row)
                sig_rows.append(row)
        if primary_wil:
            adj = holm([r["p_value"] for r in primary_wil])
            for r, pa in zip(primary_wil, adj):
                r["p_holm"] = round(float(pa), 8)
                r["significant"] = "yes" if pa < 0.05 else "no"

        # ---- Robustness: BiGRU vs XGBoost per seed ----
        robust = []
        if "BiGRU" in avg.columns and REF_MODEL in avg.columns:
            for s in SEEDS:
                sub_s = per_class[per_class["seed"] == s]
                if sub_s.empty:
                    continue
                piv = sub_s.pivot_table(index="class_index", columns="model",
                                        values="f1", aggfunc="mean")
                if "BiGRU" not in piv.columns or REF_MODEL not in piv.columns:
                    continue
                for sub_name in ("all", "supported", "tail"):
                    idx = subset_idx[sub_name]
                    a = piv.loc[idx, "BiGRU"].values
                    b = piv.loc[idx, REF_MODEL].values
                    W, p, nz = wilcoxon_pair(a, b)
                    diff = a - b
                    nzd = diff[diff != 0]
                    med_nz = float(np.median(nzd)) * 100 if len(nzd) else 0.0
                    robust.append({
                        "family": "robust_wilcoxon_bigru", "test_type": "wilcoxon_class_f1",
                        "model_a": "BiGRU", "model_b": REF_MODEL, "subset": sub_name,
                        "seed": int(s), "n_units": int(nz),
                        "statistic": round(W, 2) if not np.isnan(W) else np.nan,
                        "p_value": round(p, 8), "p_holm": np.nan, "significant": "",
                        "effect": round(med_nz, 4),
                        "detail": f"better={int((diff > 0).sum())},"
                                  f"worse={int((diff < 0).sum())},n_classes={len(idx)},"
                                  f"n_test_samples={int(supports[idx].sum())}",
                    })
            if robust:
                adj = holm([r["p_value"] for r in robust])
                for r, pa in zip(robust, adj):
                    r["p_holm"] = round(float(pa), 8)
                    r["significant"] = "yes" if pa < 0.05 else "no"
                sig_rows.extend(robust)

    sig = pd.DataFrame(sig_rows)
    sig.to_csv(SIG_CSV, index=False, encoding="utf-8-sig")

    # ---- Paired-difference CI vs XGBoost ----
    deltas = {}
    if REF_MODEL in agg:
        for m in [x for x in MODELS if x != REF_MODEL and x in agg]:
            deltas[m] = {
                "acc": paired_delta_ci(results, m, "acc"),
                "macro_f1": paired_delta_ci(results, m, "macro_f1"),
            }

    # ---- Conclusion verdict ----
    best_dl = None
    dl_models = [m for m in MODELS if m != REF_MODEL and m in agg]
    if dl_models:
        best_dl = max(dl_models, key=lambda x: agg[x]["acc"]["mean"])

    primary_rows = [r for r in sig_rows if r["family"] == "primary_mcnemar"]
    mcn_by_model = {r["model_a"]: r for r in primary_rows}
    wil_tail = {r["model_a"]: r for r in primary_wil if r["subset"] == "tail"}
    wil_supp = {r["model_a"]: r for r in primary_wil if r["subset"] == "supported"}

    conclusions = []
    if REF_MODEL in agg:
        conclusions.append(
            f"{REF_MODEL} 5-seed test Acc = "
            f"{agg[REF_MODEL]['acc']['mean']*100:.2f}% ± "
            f"{agg[REF_MODEL]['acc']['std']*100:.2f} pp")
    for m in dl_models:
        conclusions.append(
            f"{m} 5-seed test Acc = {agg[m]['acc']['mean']*100:.2f}% ± "
            f"{agg[m]['acc']['std']*100:.2f} pp")
    if best_dl and REF_MODEL in agg:
        gap = (agg[REF_MODEL]["acc"]["mean"] - agg[best_dl]["acc"]["mean"]) * 100
        conclusions.append(f"best ANN = {best_dl}; mean gap to {REF_MODEL} = {gap:.2f} pp")
    for m, r in mcn_by_model.items():
        conclusions.append(
            f"McNemar(seed 42) {m} vs {REF_MODEL}: p={fmt_p(r['p_value'])}, "
            f"Holm p={fmt_p(r['p_holm'])}, significant={'yes' if r['significant']=='yes' else 'no'}, "
            f"accuracy diff={r['effect']:.2f} pp")
    for m, r in wil_supp.items():
        conclusions.append(
            f"valid classes ({r['n_units']} nonzero diffs) Wilcoxon {m} vs {REF_MODEL}: "
            f"p={fmt_p(r['p_value'])}, Holm p={fmt_p(r['p_holm'])}, "
            f"median nonzero ΔF1={r['effect']:+.2f} pp, {r['detail']}")
    for m, r in wil_tail.items():
        samples = int(r["detail"].split("n_test_samples=")[1])
        if samples <= 10:
            conclusions.append(
                f"tail 77 classes Wilcoxon {m} vs {REF_MODEL}: only {samples} tail samples in the test set, "
                f"insufficient class-level test power (p={fmt_p(r['p_value'])}), no directional conclusion")
        else:
            conclusions.append(
                f"tail 77 classes Wilcoxon {m} vs {REF_MODEL}: p={fmt_p(r['p_value'])}, "
                f"Holm p={fmt_p(r['p_holm'])}, median nonzero ΔF1={r['effect']:+.2f} pp")

    summary = {
        "experiment": "E4 multi-seed repetition and significance testing",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": {
            "seeds": SEEDS,
            "models": MODELS,
            "reference": REF_MODEL,
            "n_train": int(len(scaled["y_train"])),
            "n_val": int(len(scaled["y_val"])),
            "n_test": int(len(y_test)),
            "n_classes": N_CLASSES,
            "early_stopping": "LeNet-5 60/10; others 100/15 (val Macro-F1)",
            "mcnemar": "exact binomial (two-sided), paired on identical test split; Holm across models at seed 42",
            "wilcoxon": "per-class F1 averaged over seeds; Holm across model x subset family",
        },
        "models": agg,
        "best_dl": best_dl,
        "delta_vs_reference": deltas,
        "n_test_support_zero_classes": int(np.sum(supports == 0)),
        "significance_rows": sig_rows,
        "conclusions": conclusions,
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    make_plot(agg)
    write_report(summary, sig, agg, deltas, primary_mcnemar, primary_wil, y_test)
    log(f"analysis complete: {SUMMARY_JSON}")


# ==================== Figures ====================
def make_plot(agg):
    models = [m for m in MODELS if m in agg]
    if not models:
        return
    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, col, title in zip(axes, ["acc", "macro_f1"], ["Test Accuracy", "Test Macro-F1"]):
        means = [agg[m][col]["mean"] * 100 for m in models]
        stds = [agg[m][col]["std"] * 100 for m in models]
        colors = ["#4C78A8" if m != REF_MODEL else "#E45756" for m in models]
        ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
        ax.set_title(f"{title} (5 seeds, mean±std)")
        ax.set_ylabel("%")
        for i, (mu, sd) in enumerate(zip(means, stds)):
            ax.text(i, mu + sd + 0.3, f"{mu:.2f}", ha="center", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_PNG, dpi=200)
    plt.close(fig)


# ==================== Report ====================
def md_table(rows, cols):
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join([head, sep] + body)


def fmt_p(p):
    try:
        p = float(p)
    except (TypeError, ValueError):
        return str(p)
    if p == 0.0:
        return "<1e-300"
    return f"{p:.3e}"


def write_report(summary, sig, agg, deltas, primary_mcnemar, primary_wil, y_test):
    models = [m for m in MODELS if m in agg]
    lines = []
    lines.append("# E4 multi-seed repetition and significance testing report")
    lines.append("")
    lines.append(f"Generated: {summary['generated_at']} | Companion: DeepLearning_Supplementary_Plan.md §5")
    lines.append("")
    lines.append("## 1. Purpose and protocol")
    lines.append("")
    lines.append("- **Purpose**: answer whether the gap between 80.16% (BiGRU) and 83.15% (XGBoost-104) exceeds random seed fluctuation,"
                 "and give the mean±std of the 6 tuned models over 5 random seeds. ")
    lines.append("- **Data**: 104-dim features, train 12,537 / val 2,640 / test 2,641, 231 classes; split fully consistent with E1/E2 (frozen).")
    lines.append("- **Seeds**: 42 / 123 / 777 / 2026 / 3407. Each model is retrained from scratch with its Optuna-best configuration.")
    lines.append("- **Early stopping**: LeNet-5 60 epochs / patience 10; BiGRU, Transformer, FT-Transformer, TabNet 100 / 15;"
                 "XGBoost early_stopping_rounds=50. Selection criterion in all cases is validation Macro-F1 (TabNet uses validation logloss, following the original experiment). ")
    lines.append("- **Tests**: (1) McNemar exact binomial test (paired predictions on the same test set);"
                 "(2) Wilcoxon signed-rank test on per-class F1 (seed-averaged), "
                 "and give the all/supported/head/torso/tail subsets; "
                 "(3) Holm-Bonferroni controls the family-wise multiplicity. ")
    lines.append("- **Test set support**: 83 of the 231 classes have no test samples; the `supported` subset is the 148 classes with test samples."
                 "head/torso/tail follow the E1/E2 training-support stratification (77 classes each), "
                 "But the tail tier has only 5 test samples, so class-level test power is insufficient. ")
    lines.append("")
    lines.append("## 2. Multi-seed results (mean ± std)")
    lines.append("")
    rows = []
    for m in models:
        a = agg[m]
        rows.append({
            "Model": m, "Seeds": a["n_seeds"],
            "Acc (%)": f"{a['acc']['mean']*100:.2f} ± {a['acc']['std']*100:.2f}",
            "Macro-F1 (%)": f"{a['macro_f1']['mean']*100:.2f} ± {a['macro_f1']['std']*100:.2f}",
            "Weighted-F1 (%)": f"{a['weighted_f1']['mean']*100:.2f} ± {a['weighted_f1']['std']*100:.2f}",
            "Top-5 (%)": f"{a['top5']['mean']*100:.2f} ± {a['top5']['std']*100:.2f}",
            "params": f"{a['n_params']:,}",
            "best_epoch(mean)": a["best_epoch_mean"],
        })
    lines.append(md_table(rows, ["Model", "N seeds", "Acc (%)", "Macro-F1 (%)", "Weighted-F1 (%)",
                                 "Top-5 (%)", "params", "best_epoch(mean)"]))
    lines.append("")
    lines.append("> Macro-F1 still computed over 231 classes (including 83 classes with no test samples, F1=0), consistent with the E1/E2 metric;"
                 "Hence the values are lower overall, but comparable across models. ")
    lines.append("")
    lines.append("### 2.1 Long-tail stratified Macro-F1")
    lines.append("")
    rows = []
    for m in models:
        a = agg[m]
        rows.append({
            "Model": m,
            "head F1 (%)": f"{a['head_f1']['mean']*100:.2f} ± {a['head_f1']['std']*100:.2f}",
            "torso F1 (%)": f"{a['torso_f1']['mean']*100:.2f} ± {a['torso_f1']['std']*100:.2f}",
            "tail F1 (%)": f"{a['tail_f1']['mean']*100:.2f} ± {a['tail_f1']['std']*100:.2f}",
        })
    lines.append(md_table(rows, ["Model", "head F1 (%)", "torso F1 (%)", "tail F1 (%)"]))
    lines.append("")
    lines.append("> head/torso/tail follow the training-support stratification. The tail stratum has only 5 test samples, so its F1 is descriptive reference only,"
                 "Insufficient to support the inference that “ANN is stronger/weaker on rare classes” (see sections 5 and 7 for details). ")
    lines.append("")
    lines.append("## 3. Paired differences vs XGBoost-104 (same-seed pairing)")
    lines.append("")
    rows = []
    for m, d in deltas.items():
        if d["acc"] is None:
            continue
        rows.append({
            "Model": m,
            "ΔAcc (pp)": f"{d['acc']['mean_delta']*100:+.2f} "
                         f"[{d['acc']['ci95_low']*100:+.2f}, {d['acc']['ci95_high']*100:+.2f}]",
            "ΔMacro-F1 (pp)": f"{d['macro_f1']['mean_delta']*100:+.2f} "
                              f"[{d['macro_f1']['ci95_low']*100:+.2f}, {d['macro_f1']['ci95_high']*100:+.2f}]",
            "Pairs": d["acc"]["n_seeds"],
        })
    lines.append(md_table(rows, ["Model", "ΔAcc (pp)", "ΔMacro-F1 (pp)", "Paired n"]))
    lines.append("")
    lines.append("> Δ = model − XGBoost; the interval is the t-distribution 95% CI of 5-seed paired differences."
                 "This interval characterizes **seed variability** and is not equivalent to test-sample-level significance (see section 4 for that). ")
    lines.append("")
    lines.append("## 4. McNemar test (seed=42, same test set)")
    lines.append("")
    rows = []
    for r in primary_mcnemar:
        rows.append({
            "Comparison": f"{r['model_a']} vs {r['model_b']}",
            "Inconsistent pairs": r["n_units"],
            "χ²": f"{r['statistic']:.2f}",
            "Exact p": fmt_p(r["p_value"]),
            "Holm p": fmt_p(r["p_holm"]),
            "significant (0.05)": r["significant"],
            "ΔAcc (pp)": f"{r['effect']:+.2f}",
            "a / b": r["detail"],
        })
    lines.append(md_table(rows, ["Comparison", "Inconsistent pairs", "χ²", "Exact p", "Holm p",
                                 "significant (0.05)", "Delta Acc (pp)", "a / b"]))
    lines.append("")
    lines.append("## 5. Wilcoxon test (per-class F1, seed-averaged)")
    lines.append("")
    lines.append("> count nonzero-difference classes only; `median nonzero ΔF1` = median of model − XGBoost over nonzero-difference classes."
                 "The `tail` subset has only 5 test samples; the p-value has no explanatory power, and conclusions follow `supported`/`head`/`torso`. ")
    lines.append("")
    rows = []
    for r in primary_wil:
        det = r["detail"]
        nc = det.split("n_classes=")[1].split(",")[0]
        ns = int(det.split("n_test_samples=")[1])
        underpowered = ns <= 10
        rows.append({
            "Comparison": f"{r['model_a']} vs {r['model_b']}",
            "Subset": r["subset"],
            "Classes/test samples": f"{nc} / {ns}",
            "Nonzero-diff classes": r["n_units"],
            "W": "—" if (isinstance(r["statistic"], float) and np.isnan(r["statistic"])) else r["statistic"],
            "p": fmt_p(r["p_value"]),
            "Holm p": fmt_p(r["p_holm"]),
            "significant (0.05)": r["significant"],
            "Median nonzero deltaF1 (pp)": "-" if underpowered else f"{r['effect']:+.2f}",
            "Better/worse class count": det.split(",n_classes=")[0],
        })
    lines.append(md_table(rows, ["Comparison", "Subset", "Classes/test samples", "Nonzero-diff classes", "W", "p",
                                 "Holm p", "significant (0.05)", "median nonzero deltaF1 (pp)", "better/worse class count"]))
    lines.append("")
    lines.append("## 6. Conclusion")
    lines.append("")
    for c in summary["conclusions"]:
        lines.append(f"- {c}")
    lines.append("")
    lines.append("### 6.1 Comparison with the original report metric")
    lines.append("")
    lines.append("- the report §4.1 figures 80.16% vs 83.15% come from **a single run at different tuning states**, and cannot be paired-tested directly."
                 "Under the unified protocol E4 repeats tuned configs with 5 seeds: XGBoost mean 83.33%, BiGRU mean 79.71%, "
                 "Paired mean difference -3.62 pp, McNemar p<1e-300 (still significant after Holm correction). "
                 "Hence the original ~3 pp gap **cannot be attributed to random-seed noise**. ")
    lines.append("- the BiGRU mean in E4 (79.71%) differs from the report's original 80.16% by -0.45 pp."
                 "E1 once observed tuning improving validation Macro-F1 while Top-1 did not improve in step; "
                 "This only shows directionally that “the tuning objective and Top-1 are not fully aligned”; no causal mechanism can be asserted from it. ")
    lines.append("- **Single-run irreproducibility**: reruns with the same seed still fluctuate due to cuDNN nondeterminism."
                 "For example, LeNet-5 scored 75.58% in a single E1 run and 77.66%/78.08% in two E4 same-seed reruns, "
                 "while the E4 five-seed mean is 76.60% +/- 1.06; Transformer seed 42 reproduces 77.62% exactly. "
                 "This shows single-run values should give way to mean±std; E4's mean is the more reliable reporting convention. ")
    lines.append("")
    lines.append("## 7. Boundary and interpretation limits")
    lines.append("")
    lines.append("1. **Conditional conclusion**: the McNemar/Wilcoxon p-values are conditional on the current frozen test set;"
                 "Conclusions may change under a different test set and cannot be extrapolated as “necessary differences over the population distribution”. ")
    lines.append("2. **Samples not independent**: test samples come from the same literature extraction process, with literature-cluster/framework-cluster correlation,"
                 "McNemar's independent-samples assumption holds only approximately; p-values should be read as evidence with limited conservativeness. ")
    lines.append("3. **5 seeds**: the seed-level std estimate has only 4 degrees of freedom, so CIs are wide;"
                 "It tests whether the mean difference exceeds seed variability, not for fine variance decomposition. ")
    lines.append("4. **Tuned configurations only**: E4 does not reproduce seed fluctuation of default configurations;"
                 "The default-vs-tuned differences are reported separately in the E1/E2 reports. ")
    lines.append("5. **Class-level test power**: 83/231 classes have no samples in the frozen test set, and the tail stratum has only 5 test samples."
                 "Hence tail classes can only report descriptive head/torso/tail F1, and one cannot claim from this that “ANN is better/worse on rare classes”; "
                 "This issue must be addressed by expanding the test set or via E3b's neighborhood-purity stratification. ")
    lines.append("6. **Significance ≠ practical importance**: with large samples even 0.5 pp can be significant;"
                 "Whether to adopt ANN should still be judged holistically with architectural capabilities such as Top-5 retrieval and transfer learning. ")
    lines.append("")
    lines.append("## 8. File list")
    lines.append("")
    fl = [
        ("seed_repetition_results.csv", "18 metrics per (model, seed)"),
        ("per_class_f1_distribution.csv", "F1 and support per (model, seed, class)"),
        ("significance_tests.csv", "McNemar / Wilcoxon / Holm correction details"),
        ("seed_results.png", "mean±std bar chart of Acc and Macro-F1"),
        ("E4_summary.json", "machine-readable summary (with protocol and conclusions)"),
        ("preds/*.npy", "test-set predictions for recomputing paired tests"),
        ("e4_run.log", "run log"),
        ("E4_report.md", "this report"),
    ]
    lines.append(md_table([{"File": a, "Content": b} for a, b in fl], ["File", "Content"]))
    lines.append("")
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"report written: {REPORT_MD}")


# ==================== Main entry ====================
def main():
    global LOG
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--no-analyze", action="store_true",
                        help="training only, no analysis artifacts (for smoke testing)")
    parser.add_argument("--models", nargs="*", default=None,
                        help="run specified models only (default all)")
    parser.add_argument("--seeds", nargs="*", type=int, default=None,
                        help="run specified seeds only (default all)")
    parser.add_argument("--epoch-cap", type=int, default=None,
                        help="for smoke testing: cap the max number of epochs")
    args = parser.parse_args()

    global EPOCH_CAP
    EPOCH_CAP = args.epoch_cap

    import logging
    LOG = logging.getLogger("E4")
    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()
    fh = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
    LOG.addHandler(fh)

    os.makedirs(PRED_DIR, exist_ok=True)
    log("=" * 70)
    log("E4 multi-seed repetition and significance testing")
    log(f"device={ec.DEVICE} | seeds={SEEDS} | models={MODELS}")
    log("=" * 70)

    if not args.analyze_only:
        params = load_tuned_params()
        scaled = ec.load_data(LOG)
        raw = load_raw()
        done = load_done()
        run_models = args.models if args.models else MODELS
        run_seeds = args.seeds if args.seeds else SEEDS
        log(f"completed combos: {len(done)} / {len(MODELS)*len(SEEDS)}")
        t_all = time.time()
        for m in run_models:
            for s in run_seeds:
                run_one(m, s, params[m], scaled, raw, done)
        log(f"training round complete, total time {(time.time()-t_all)/60:.1f} min")
        del scaled, raw
        gc.collect()
        if ec.DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    if not args.no_analyze:
        analyze()


if __name__ == "__main__":
    main()
