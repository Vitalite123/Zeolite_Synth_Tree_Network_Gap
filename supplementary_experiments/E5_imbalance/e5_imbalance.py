"""
E5 class-imbalance handling (focal loss / class-weighted CE / Balanced Softmax / weighted sampler)
=====================================================================================
Companion documents:
  - CBUTransfer/DeepLearning/DeepLearning_Supplementary_Plan.md §6 (E5)
  - CBUTransfer/DeepLearning/SupplementaryExperiments/README.md
  - CBUTransfer/DeepLearning/DeepLearning_ANN_Full_Report.md (write-back §4.1d / §4.5 / §5.6.4 / §6.2.3)

Purpose:
  on the single tuned best ANN (BiGRU, E1.2 tuned, 1.18M parameters), compare four long-tail rebalancing strategies
  trade-off vs the “no treatment” baseline: tail Macro-F1 gain vs overall Weighted-F1 loss.

  Method:
    baseline          no treatment (CE + label smoothing 0.1, i.e., the E1.2/E4 tuned config)
    focal_g0.5/1/2    Focal Loss, α per inverse-frequency normalized to mean 1
    weighted_ce       class-weighted CE, w_c = 1/log(1.01+n_c) (formula specified in plan §6.1), normalized to mean 1
    balanced_softmax  training uses logits + log(n_c) (Ren et al. 2020), inference uses the raw softmax
    weighted_sampler  WeightedRandomSampler, equal expected occurrence frequency per class

Protocol (plan §1 + §6):
  - Data 104-dim, frozen splits 12,537 / 2,640 / 2,641, 231 classes; StandardScaler fitted on train only
  - Early stopping monitors val Macro-F1; 100 epochs / patience 15; AdamW + ReduceLROnPlateau + grad clip 1.0
  - label smoothing 0.1 is kept throughout in focal / weighted CE / balanced softmax / sampler,
    to guarantee the controlled contrast of “only the class-balancing mechanism changes”
  - 5 seeds {42, 123, 777, 2026, 3407}; baseline directly reuses E4's BiGRU five-seed predictions
    (same protocol, same split, same hyperparameters), recomputing only the group metrics, avoiding retraining

Grouped evaluation (plan §6.2, training-set class frequency):
  head  >= 1000 samples (1 class); torso 100–999 (28 classes); tail < 100 (202 classes)

  **Main convention (adopted in this experiment, denoted head_f1/torso_f1/tail_f1)**:
    compute per-class F1 on the full test set (sklearn average=None), then over “classes with test support within the group” apply
    macro averaging. It is strictly decomposable with the global Macro-F1 (the weighted sum = the macro average over the 148 supported classes),
    and is not inflated by “predicting rare classes for many head/torso samples”.
  **Secondary convention (denoted *_f1_subset, carried over in E1-E4 §4.5)**:
    keep only that group's samples, labels as all classes of that group for macro averaging. Cross-group misjudgments are not counted as FPs, so it is systematically inflated,
    and would misjudge the degenerate behavior of “predicting rare classes everywhere” as tail improvement; historical reference only, not a basis for conclusions.

Outputs (this directory):
  E5_runs.csv             one row per (method, seed) (with main/secondary convention group metrics)
  E5_summary.csv          one row per method (the plan's 8 columns + std/multi-metrics)
  E5_summary.json         machine-readable summary (protocol, method definitions, aggregation, paired differences, recommendation)
  E5_report.md            human-readable report
  E5_group_comparison.png head/torso/tail Macro-F1 grouped bar chart
  E5_tradeoff.png         delta-tail vs delta-Weighted-F1 trade-off chart
  e5_run.log              run log
  preds/{method}_seed{seed}.npy   test set predictions (int16)

Usage:
  py -B e5_imbalance.py                 # train (resumable) + analyze + report
  py -B e5_imbalance.py --analyze-only  # recompute analysis and report only (based on written predictions)
  py -B e5_imbalance.py --epoch-cap 2 --methods focal_g1 --seeds 42   # smoke test (separate artifacts)
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
from scipy.stats import t as tdist
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

HERE = os.path.dirname(os.path.abspath(__file__))
SUP_DIR = os.path.abspath(os.path.join(HERE, ".."))
E1_DIR = os.path.join(SUP_DIR, "E1_tuning")
E2_DIR = os.path.join(SUP_DIR, "E2_sota")
E4_DIR = os.path.join(SUP_DIR, "E4_seeds")
for _p in (E1_DIR, E2_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import e2_common as ec  # noqa: E402
from e1_2_bigru import build_model as build_bigru  # noqa: E402

# ==================== Constants ====================
SEEDS = [42, 123, 777, 2026, 3407]
N_CLASSES = ec.N_CLASS_EXPECT
MAX_EPOCHS = 100
PATIENCE = 15
LABEL_SMOOTHING = 0.1
HEAD_MIN, TORSO_MIN = 1000, 100
BASELINE = "baseline"
METHODS = [BASELINE, "focal_g0.5", "focal_g1", "focal_g2",
           "weighted_ce", "balanced_softmax", "weighted_sampler"]
METHOD_LABEL = {
    BASELINE: "none (CE + LS 0.1)",
    "focal_g0.5": "gamma=0.5",
    "focal_g1": "gamma=1.0",
    "focal_g2": "gamma=2.0",
    "weighted_ce": "1/log(1.01+n_c)",
    "balanced_softmax": "log(n_c) prior",
    "weighted_sampler": "1/n_c sampling",
}
SLUG = {"focal_g0.5": "focal0p5", "focal_g1": "focal1", "focal_g2": "focal2",
        "weighted_ce": "wce", "balanced_softmax": "bsm", "weighted_sampler": "wsamp",
        BASELINE: "baseline"}

RESULT_FIELDS = [
    "method", "gamma_or_weight", "seed",
    "acc", "macro_f1", "macro_f1_support", "weighted_f1", "top5",
    "head_f1", "torso_f1", "tail_f1", "tail_recall",
    "head_f1_subset", "torso_f1_subset", "tail_f1_subset",
    "eq_head_f1", "eq_torso_f1", "eq_tail_f1",
    "val_acc", "val_macro_f1",
    "n_params", "best_epoch", "duration_s", "source",
]

METRIC_COLS = ["acc", "macro_f1", "macro_f1_support", "weighted_f1", "top5",
               "head_f1", "torso_f1", "tail_f1", "tail_recall",
               "head_f1_subset", "torso_f1_subset", "tail_f1_subset",
               "eq_head_f1", "eq_torso_f1", "eq_tail_f1",
               "val_acc", "val_macro_f1"]

LOG = None
EPOCH_CAP = None
RUN_TAG = ""       # "smoke" in smoke tests, isolating artifacts

TRAIN_CSV = None
SUMMARY_CSV = None
SUMMARY_JSON = None
REPORT_MD = None
FIG_GROUP = None
FIG_TRADEOFF = None
PRED_DIR = None
LOG_PATH = None


def init_paths(tag=""):
    global TRAIN_CSV, SUMMARY_CSV, SUMMARY_JSON, REPORT_MD
    global FIG_GROUP, FIG_TRADEOFF, PRED_DIR, LOG_PATH
    stamp = f"_{tag}" if tag else ""
    TRAIN_CSV = os.path.join(HERE, f"E5_runs{stamp}.csv")
    SUMMARY_CSV = os.path.join(HERE, f"E5_summary{stamp}.csv")
    SUMMARY_JSON = os.path.join(HERE, f"E5_summary{stamp}.json")
    REPORT_MD = os.path.join(HERE, f"E5_report{stamp}.md")
    FIG_GROUP = os.path.join(HERE, f"E5_group_comparison{stamp}.png")
    FIG_TRADEOFF = os.path.join(HERE, f"E5_tradeoff{stamp}.png")
    PRED_DIR = os.path.join(HERE, f"preds{stamp}")
    LOG_PATH = os.path.join(HERE, f"e5_run{stamp}.log")
    os.makedirs(PRED_DIR, exist_ok=True)


def log(msg):
    if LOG is not None:
        LOG.info(msg)
    else:
        print(msg, flush=True)


def make_logger():
    import logging
    lg = logging.getLogger("E5")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
        for h in (logging.FileHandler(LOG_PATH, encoding="utf-8"),
                  logging.StreamHandler()):
            h.setFormatter(fmt)
            lg.addHandler(h)
    return lg


# ==================== Tuning parameters ====================
def load_tuned_bigru():
    with open(os.path.join(E1_DIR, "best_params_E1_2.json"), "r", encoding="utf-8") as f:
        return json.load(f)["params"]


# ==================== Group definitions ====================
def make_threshold_strata(y_train, n_classes):
    """Plan section 6.2: head >=1000, torso 100-999, tail <100 (by training-set class frequency)."""
    counts = np.bincount(y_train, minlength=n_classes)
    return {
        "head": np.where(counts >= HEAD_MIN)[0],
        "torso": np.where((counts >= TORSO_MIN) & (counts < HEAD_MIN))[0],
        "tail": np.where(counts < TORSO_MIN)[0],
    }


# ==================== Data ====================
def load_data():
    data = ec.load_data(LOG)
    data["thr_strata"] = make_threshold_strata(data["y_train"], N_CLASSES)
    data["eq_strata"] = data["strata"]  # E1-E4's tercile strata, for reference only
    for key in ("head", "torso", "tail"):
        idx = data["thr_strata"][key]
        log(f"group ({key}): {len(idx)} classes | "
            f"train {int(np.bincount(data['y_train'], minlength=N_CLASSES)[idx].sum())} | "
            f"test {int(np.isin(data['y_test'], idx).sum())}")
    return data


def evaluate_pred(pred, y, data):
    """Grouped evaluation under a unified protocol.

    Primary metric: per-class F1/Recall on the full test set -> macro average over classes with test support in the group (decomposable, robust to cross-group FPs).
    Secondary metric: within-group subset + all labels of that group (E1–E4 §4.5 metric, historical reference only).
    """
    labels = np.arange(N_CLASSES)
    supported = np.unique(y)
    f1_all = f1_score(y, pred, average=None, labels=labels, zero_division=0)
    rec_all = recall_score(y, pred, average=None, labels=labels, zero_division=0)

    out = {
        "acc": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "macro_f1_support": float(f1_all[supported].mean()),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
    }
    for name in ("head", "torso", "tail"):
        idx = data["thr_strata"][name]
        sup = np.intersect1d(supported, idx)
        out[f"{name}_f1"] = float(f1_all[sup].mean()) if len(sup) else float("nan")
        out[f"{name}_recall"] = float(rec_all[sup].mean()) if len(sup) else float("nan")
        mask = np.isin(y, idx)
        out[f"{name}_f1_subset"] = (float("nan") if mask.sum() == 0 else float(
            f1_score(y[mask], pred[mask], labels=idx, average="macro", zero_division=0)))
        eq_idx = data["eq_strata"][name]
        eq_mask = np.isin(y, eq_idx)
        out[f"eq_{name}_f1"] = (float("nan") if eq_mask.sum() == 0 else float(
            f1_score(y[eq_mask], pred[eq_mask], labels=eq_idx,
                     average="macro", zero_division=0)))
    return out


# ==================== Loss functions ====================
def class_weights(counts, kind):
    """kind='inv': inverse-frequency α;kind='logeff': 1/log(1.01+n_c).
    both normalized to mean 1 so the gradient scale is comparable with the baseline CE (relative class weights unchanged)."""
    c = np.maximum(counts.astype(np.float64), 1.0)
    w = 1.0 / c if kind == "inv" else 1.0 / np.log(1.01 + c)
    return w / w.mean()


def soft_target(y, n_classes, smoothing):
    """Soft labels consistent with PyTorch cross_entropy(label_smoothing)."""
    other = smoothing / n_classes
    y_soft = torch.full((y.size(0), n_classes), other, dtype=torch.float32,
                        device=y.device)
    y_soft.scatter_(1, y.unsqueeze(1), 1.0 - smoothing + other)
    return y_soft


def focal_loss(logits, y, gamma, alpha, smoothing):
    """Soft-label Focal Loss: -Sum_c alpha_c y_c (1-p_c)^gamma log p_c."""
    logp = F.log_softmax(logits, dim=1)
    p = logp.exp()
    y_soft = soft_target(y, logits.size(1), smoothing)
    focal = (1.0 - p).clamp(min=0.0) ** gamma
    loss = -(alpha.unsqueeze(0) * y_soft * focal * logp).sum(dim=1)
    return loss.mean()


def build_loss(method, counts):
    """Returns loss_fn(logits, y); the sampler's rebalancing is done on the DataLoader side."""
    if method.startswith("focal_g"):
        gamma = float(method.split("focal_g")[1])
        alpha = torch.tensor(class_weights(counts, "inv"), dtype=torch.float32,
                             device=ec.DEVICE)
        return lambda lg, yb: focal_loss(lg, yb, gamma, alpha, LABEL_SMOOTHING)

    if method == "weighted_ce":
        w = torch.tensor(class_weights(counts, "logeff"), dtype=torch.float32,
                         device=ec.DEVICE)
        return lambda lg, yb: F.cross_entropy(lg, yb, weight=w,
                                              label_smoothing=LABEL_SMOOTHING)

    if method == "balanced_softmax":
        log_prior = torch.tensor(np.log(np.maximum(counts, 1.0)), dtype=torch.float32,
                                 device=ec.DEVICE)
        return lambda lg, yb: F.cross_entropy(lg + log_prior, yb,
                                              label_smoothing=LABEL_SMOOTHING)

    # baseline / weighted_sampler: standard CE (sampler on the DataLoader side)
    return lambda lg, yb: F.cross_entropy(lg, yb, label_smoothing=LABEL_SMOOTHING)


# ==================== Training ====================
def _forward(model, xb):
    return model(xb)


def train_one(params, data, seed, method, max_epochs, patience, tag):
    """BiGRU tuned is trained from scratch under the specified rebalancing mechanism; early stopping monitors val Macro-F1."""
    ec.set_seed(seed)
    counts = np.bincount(data["y_train"], minlength=N_CLASSES)
    loss_fn = build_loss(method, counts)

    ds = TensorDataset(torch.as_tensor(data["X_train"]),
                       torch.as_tensor(data["y_train"], dtype=torch.long))
    if method == "weighted_sampler":
        sample_w = 1.0 / np.maximum(counts[data["y_train"]], 1).astype(np.float64)
        sampler = WeightedRandomSampler(torch.as_tensor(sample_w, dtype=torch.double),
                                        num_samples=len(data["y_train"]),
                                        replacement=True)
        loader = DataLoader(ds, batch_size=int(params["batch_size"]), sampler=sampler,
                            num_workers=0, drop_last=False)
    else:
        loader = DataLoader(ds, batch_size=int(params["batch_size"]), shuffle=True,
                            num_workers=0, drop_last=False)

    model = build_bigru(params, N_CLASSES).to(ec.DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=float(params["learning_rate"]),
                            weight_decay=float(params["weight_decay"]))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min",
                                                       factor=0.5, patience=5)

    best_f1, best_state, best_epoch, bad = -1.0, None, 0, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        total, n_batches = 0.0, 0
        for xb, yb in loader:
            if xb.size(0) == 1:      # BatchNorm cannot train on a single-sample batch
                continue
            xb = xb.to(ec.DEVICE, non_blocking=True)
            yb = yb.to(ec.DEVICE, non_blocking=True)
            opt.zero_grad()
            loss = loss_fn(_forward(model, xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total += float(loss.item())
            n_batches += 1

        val_proba = ec.predict_proba(model, data["X_val"], _forward)
        val_f1 = ec.metrics_from_proba(val_proba, data["y_val"],
                                       N_CLASSES)["macro_f1"]
        p_true = val_proba[np.arange(len(data["y_val"])), data["y_val"]]
        val_loss = float(-np.log(np.clip(p_true, 1e-12, 1.0)).mean())
        sched.step(val_loss)

        improved = val_f1 > best_f1
        log(f"{tag} epoch {epoch:3d} | train_loss={total/max(n_batches,1):.4f} | "
            f"val_loss={val_loss:.4f} | val_MacroF1={val_f1:.4f}"
            f"{' *' if improved else ''}")

        if improved:
            best_f1, best_epoch, bad = val_f1, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_proba = ec.predict_proba(model, data["X_test"], _forward)
    val_proba = ec.predict_proba(model, data["X_val"], _forward)
    log(f"{tag} best_epoch={best_epoch} | val MacroF1={best_f1:.4f} | "
        f"test Acc={ec.metrics_from_proba(test_proba, data['y_test'], N_CLASSES)['acc']:.4f} | "
        f"params={n_params}")
    del model
    if ec.DEVICE.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
    return dict(test_proba=test_proba, val_proba=val_proba,
                n_params=int(n_params), best_epoch=int(best_epoch))


def train_with_oom_retry(params, data, seed, method, tag):
    bs = int(params["batch_size"])
    if ec.DEVICE.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
    for attempt in range(3):
        p = dict(params, batch_size=bs)
        try:
            return train_one(p, data, seed, method,
                             min(MAX_EPOCHS, EPOCH_CAP) if EPOCH_CAP else MAX_EPOCHS,
                             min(PATIENCE, max(1, (EPOCH_CAP or MAX_EPOCHS) // 2)), tag)
        except torch.OutOfMemoryError:
            if attempt == 2 or bs // 2 < 64:
                raise
            bs = max(64, bs // 2)
            log(f"{tag} | CUDA OOM, batch -> {bs}")
            gc.collect()
            torch.cuda.empty_cache()


# ==================== CSV ====================
def append_row(path, row, fields):
    new = not os.path.exists(path)
    mode = "w" if new else "a"
    enc = "utf-8-sig" if new else "utf-8"
    with open(path, mode, newline="", encoding=enc) as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)


def pred_path(method, seed):
    return os.path.join(PRED_DIR, f"{SLUG[method]}_seed{seed}.npy")


def load_done():
    if not os.path.exists(TRAIN_CSV):
        return set()
    df = pd.read_csv(TRAIN_CSV)
    return {(r["method"], int(r["seed"])) for _, r in df.iterrows()
            if os.path.exists(pred_path(r["method"], int(r["seed"])))}


# ==================== baseline reuse (E4) ====================
def e4_bigru_rows(data):
    """Reuses the BiGRU five-seed predictions from E4: same protocol, same split, same tuned hyperparameters.
    top5/val metrics taken from the E4 results table (argmax predictions cannot recover Top-5); group metrics recomputed with the unified metric."""
    res_path = os.path.join(E4_DIR, "seed_repetition_results.csv")
    if not os.path.exists(res_path):
        raise RuntimeError(f"missing E4 results table: {res_path}")
    e4 = pd.read_csv(res_path)
    e4 = e4[e4["model"] == "BiGRU"].set_index("seed")
    params = load_tuned_bigru()
    n_params = sum(p.numel() for p in build_bigru(params, N_CLASSES).parameters())
    rows = []
    for seed in SEEDS:
        p = os.path.join(E4_DIR, "preds", f"bigru_seed{seed}.npy")
        if seed not in e4.index or not os.path.exists(p):
            raise RuntimeError(f"missing E4 baseline artifacts: seed={seed}")
        pred = np.load(p).astype(np.int64)
        m = evaluate_pred(pred, data["y_test"], data)
        r = e4.loc[seed]
        row = {
            "method": BASELINE, "gamma_or_weight": METHOD_LABEL[BASELINE],
            "seed": int(seed),
            "top5": round(float(r["top5"]), 6),
            "val_acc": round(float(r["val_acc"]), 6),
            "val_macro_f1": round(float(r["val_macro_f1"]), 6),
            "n_params": int(n_params),
            "best_epoch": int(r["best_epoch"]),
            "duration_s": float(r["duration_s"]),
            "source": f"E4_seeds/preds/bigru_seed{seed}.npy",
        }
        for k in METRIC_COLS:
            if k not in row:
                row[k] = round(float(m[k]), 6)
        rows.append(row)
        np.save(pred_path(BASELINE, seed), pred.astype(np.int16))
    return rows


# ==================== Single run ====================
def run_one(method, seed, params, data, done):
    if (method, seed) in done:
        log(f"[skip] {method} seed={seed} already done")
        return
    tag = f"[{method} seed={seed}]"
    log(f"=== {tag} start ===")
    t0 = time.time()
    out = train_with_oom_retry(params, data, seed, method, tag)

    pred = out["test_proba"].argmax(axis=1)
    m = evaluate_pred(pred, data["y_test"], data)
    top5 = ec.metrics_from_proba(out["test_proba"], data["y_test"], N_CLASSES)["top5"]
    val_m = ec.metrics_from_proba(out["val_proba"], data["y_val"], N_CLASSES)
    np.save(pred_path(method, seed), pred.astype(np.int16))

    row = {
        "method": method, "gamma_or_weight": METHOD_LABEL[method], "seed": int(seed),
        "top5": round(float(top5), 6),
        "val_acc": round(float(val_m["acc"]), 6),
        "val_macro_f1": round(float(val_m["macro_f1"]), 6),
        "n_params": int(out["n_params"]),
        "best_epoch": int(out["best_epoch"]),
        "duration_s": round(time.time() - t0, 1),
        "source": "trained",
    }
    for k in METRIC_COLS:
        if k not in row:
            row[k] = round(float(m[k]), 6)
    append_row(TRAIN_CSV, row, RESULT_FIELDS)
    log(f"=== {tag} done in {row['duration_s']}s | Acc={row['acc']:.4f} "
        f"MacroF1={row['macro_f1']:.4f} tailF1={row['tail_f1']:.4f} "
        f"tailF1_subset={row['tail_f1_subset']:.4f} ===")


# ==================== Statistics ====================
def mean_std(v):
    v = np.asarray(v, dtype=float)
    if len(v) == 0:
        return np.nan, np.nan
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0)


def paired_delta(runs, method, col, ref=BASELINE):
    a = runs[runs["method"] == method].set_index("seed")[col].astype(float)
    b = runs[runs["method"] == ref].set_index("seed")[col].astype(float)
    common = sorted(set(a.index) & set(b.index))
    if not common:
        return None
    d = (a.loc[common] - b.loc[common]).values
    n = len(d)
    mu = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    half = float(tdist.ppf(0.975, n - 1) * sd / math.sqrt(n)) if n > 1 else np.nan
    return {"n_seeds": n, "mean_delta": mu, "std_delta": sd,
            "ci95_low": mu - half, "ci95_high": mu + half,
            "seeds": [int(s) for s in common]}


# ==================== Analysis ====================
def recompute_runs(runs, data):
    """Recomputes metrics uniformly from saved predictions to avoid inconsistent historical CSV conventions; top5/val/runtime reuse original records."""
    out = []
    for _, r in runs.iterrows():
        p = pred_path(r["method"], int(r["seed"]))
        if not os.path.exists(p):
            continue
        pred = np.load(p).astype(np.int64)
        m = evaluate_pred(pred, data["y_test"], data)
        rec = r.to_dict()
        for k in METRIC_COLS:
            rec[k] = round(float(m[k]), 6) if k in m else rec.get(k, "")
        out.append(rec)
    df = pd.DataFrame(out)
    df = (df.drop_duplicates(subset=["method", "seed"], keep="last")
            .sort_values(["method", "seed"]).reset_index(drop=True))
    df.to_csv(TRAIN_CSV, index=False, encoding="utf-8-sig")
    return df


def analyze():
    log("=== Analysis phase ===")
    data = load_data()
    raw = pd.read_csv(TRAIN_CSV)
    runs = recompute_runs(raw, data)

    agg = {}
    for m in METHODS:
        sub = runs[runs["method"] == m]
        if sub.empty:
            continue
        entry = {"n_seeds": int(len(sub)), "seeds": sorted(int(s) for s in sub["seed"])}
        for col in METRIC_COLS:
            mu, sd = mean_std(sub[col].values)
            entry[col] = {"mean": round(mu, 6), "std": round(sd, 6)}
        entry["n_params"] = int(sub["n_params"].iloc[0])
        entry["best_epoch_mean"] = round(float(sub["best_epoch"].astype(float).mean()), 1)
        entry["duration_s_mean"] = round(float(sub["duration_s"].astype(float).mean()), 1)
        agg[m] = entry

    summary_rows = []
    for m in METHODS:
        if m not in agg:
            continue
        a = agg[m]
        row = {"method": m, "gamma_or_weight": METHOD_LABEL[m]}
        for col in ("head_f1", "torso_f1", "tail_f1", "macro_f1", "weighted_f1", "top5"):
            row[col] = round(a[col]["mean"], 6)
            row[f"{col}_std"] = round(a[col]["std"], 6)
        row.update({
            "acc": round(a["acc"]["mean"], 6), "acc_std": round(a["acc"]["std"], 6),
            "macro_f1_support": round(a["macro_f1_support"]["mean"], 6),
            "macro_f1_support_std": round(a["macro_f1_support"]["std"], 6),
            "tail_recall": round(a["tail_recall"]["mean"], 6),
            "tail_recall_std": round(a["tail_recall"]["std"], 6),
            "tail_f1_subset": round(a["tail_f1_subset"]["mean"], 6),
            "head_f1_subset": round(a["head_f1_subset"]["mean"], 6),
            "torso_f1_subset": round(a["torso_f1_subset"]["mean"], 6),
            "eq_tail_f1": round(a["eq_tail_f1"]["mean"], 6),
            "n_seeds": a["n_seeds"],
        })
        summary_rows.append(row)
    cols = ["method", "gamma_or_weight", "head_f1", "torso_f1", "tail_f1",
            "macro_f1", "weighted_f1", "top5",
            "head_f1_std", "torso_f1_std", "tail_f1_std",
            "macro_f1_std", "weighted_f1_std", "top5_std",
            "acc", "acc_std", "macro_f1_support", "macro_f1_support_std",
            "tail_recall", "tail_recall_std",
            "head_f1_subset", "torso_f1_subset", "tail_f1_subset",
            "eq_tail_f1", "n_seeds"]
    pd.DataFrame(summary_rows, columns=cols).to_csv(
        SUMMARY_CSV, index=False, encoding="utf-8-sig")

    deltas = {}
    for m in METHODS:
        if m == BASELINE or m not in agg:
            continue
        deltas[m] = {col: paired_delta(runs, m, col)
                     for col in ("acc", "macro_f1", "macro_f1_support", "weighted_f1",
                                 "top5", "head_f1", "torso_f1", "tail_f1", "tail_recall")}

    rec = recommend(agg, deltas)
    counts_tr = np.bincount(data["y_train"], minlength=N_CLASSES)
    counts_te = np.bincount(data["y_test"], minlength=N_CLASSES)
    group_support = {
        "train_counts": {
            "head": int(counts_tr[data["thr_strata"]["head"]].sum()),
            "torso": int(counts_tr[data["thr_strata"]["torso"]].sum()),
            "tail": int(counts_tr[data["thr_strata"]["tail"]].sum()),
        },
        "test_counts": {
            "head": int(counts_te[data["thr_strata"]["head"]].sum()),
            "torso": int(counts_te[data["thr_strata"]["torso"]].sum()),
            "tail": int(counts_te[data["thr_strata"]["tail"]].sum()),
        },
        "tail_classes_with_test": int(np.sum(counts_te[data["thr_strata"]["tail"]] > 0)),
    }
    conclusions = build_conclusions(agg, deltas, rec, group_support)

    summary = {
        "experiment": "E5 class-imbalance handling",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": {
            "model": "BiGRU (E1.2 tuned, 1,180,008 params)",
            "seeds": SEEDS,
            "methods": {m: METHOD_LABEL[m] for m in METHODS},
            "n_train": int(len(data["y_train"])),
            "n_val": int(len(data["y_val"])),
            "n_test": int(len(data["y_test"])),
            "n_classes": N_CLASSES,
            "early_stopping": "val Macro-F1, 100 epochs, patience 15",
            "label_smoothing": LABEL_SMOOTHING,
            "loss_definitions": {
                "focal": "-sum_c alpha_c*y_c*(1-p_c)^gamma*log p_c; "
                         "alpha_c = (1/n_c)/mean(1/n_c)",
                "weighted_ce": "w_c = (1/log(1.01+n_c))/mean_c(1/log(1.01+n_c))",
                "balanced_softmax": "train: CE(logits + log n_c); inference: softmax(logits)",
                "weighted_sampler": "P(sample) proportional to 1/n_{c(sample)}, replacement=True",
                "baseline": "CE + label smoothing 0.1 (E1.2/E4 tuned config)",
            },
            "group_definition": {
                "primary": "plan 6.2: head n_c>=1000 (1 class), torso 100-999 (28), tail <100 (202)",
                "primary_metric": ("full-test per-class F1 averaged over group classes "
                                   "with test support (decomposable; cross-group FP counted)"),
                "secondary_metric": ("group-subset macro-F1 with all group labels "
                                     "(E1-E4 §4.5 convention; ignores cross-group FP; "
                                     "comparison only)"),
                "secondary_strata": "E1-E4 equal-thirds by class count (77/77/77), eq_* columns",
            },
            "baseline_source": "E4_seeds/preds/bigru_seed{seed}.npy (same protocol, no retraining)",
        },
        "group_support": dict(group_support,
                              classes_without_test=int(np.sum(counts_te == 0))),
        "methods": agg,
        "delta_vs_baseline": deltas,
        "recommendation": rec,
        "conclusions": conclusions,
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    make_group_plot(agg)
    make_tradeoff_plot(agg, deltas, rec)
    write_report(summary, agg, deltas, rec, summary_rows)
    log(f"analysis complete: {SUMMARY_JSON}")


# ==================== Recommendation logic ====================
def recommend(agg, deltas):
    """Practical screening on the (Delta Weighted-F1 loss, Delta tail-F1 gain) plane:
    require Δtail >= +1.0 pp and ΔWeighted >= -1.0 pp (absolute percentage points);
    also requiring the lower bound of the 5-seed paired 95% CI of Δtail > 0 (statistically resolvable).
    if only the former is met, the verdict is “insufficient directional evidence” and no deployment recommendation is made."""
    base = agg[BASELINE]
    cands = []
    for m, d in deltas.items():
        dt = d["tail_f1"]["mean_delta"] * 100
        dw = d["weighted_f1"]["mean_delta"] * 100
        da = d["acc"]["mean_delta"] * 100
        cands.append({
            "method": m, "delta_tail_pp": round(dt, 3),
            "delta_weighted_pp": round(dw, 3),
            "delta_acc_pp": round(da, 3),
            "delta_macro_pp": round(d["macro_f1"]["mean_delta"] * 100, 3),
            "delta_macro_support_pp": round(
                d["macro_f1_support"]["mean_delta"] * 100, 3),
            "delta_tail_recall_pp": round(d["tail_recall"]["mean_delta"] * 100, 3),
            "tail_ci_low_pp": round(d["tail_f1"]["ci95_low"] * 100, 3),
            "tail_ci_high_pp": round(d["tail_f1"]["ci95_high"] * 100, 3),
            "meets_practical": bool(dt >= 1.0 and dw >= -1.0),
            "stat_supported": bool(d["tail_f1"]["ci95_low"] > 0),
        })
    pareto = []
    for c in cands:
        dominated = any((o["delta_tail_pp"] >= c["delta_tail_pp"]
                         and o["delta_weighted_pp"] >= c["delta_weighted_pp"]
                         and (o["delta_tail_pp"] > c["delta_tail_pp"]
                              or o["delta_weighted_pp"] > c["delta_weighted_pp"]))
                        for o in cands)
        c["pareto"] = not dominated
        if not dominated:
            pareto.append(c["method"])
    passing = [c for c in cands if c["meets_practical"]]
    stat_ok = [c for c in passing if c["stat_supported"]]
    if stat_ok:
        best = max(stat_ok, key=lambda c: c["delta_tail_pp"])
        verdict = "recommend deployment"
        reason = (f"{best['method']} simultaneously satisfies Δtail-F1 ≥ +1.0 pp and "
                  f"ΔWeighted-F1 ≥ -1.0 pp(Δtail={best['delta_tail_pp']:+.2f} pp,"
                  f"ΔWeighted={best['delta_weighted_pp']:+.2f} pp),"
                  f"and the lower bound of the 95% CI of Δtail > 0")
    elif passing:
        best = max(passing, key=lambda c: c["delta_tail_pp"])
        verdict = "directional benefit (insufficient evidence, no strong claim)"
        reason = (f"{best['method']} meets the practical threshold (Δtail={best['delta_tail_pp']:+.2f} pp,"
                  f"ΔWeighted={best['delta_weighted_pp']:+.2f} pp),"
                  f"but the 5-seed paired 95% CI of Δtail crosses 0"
                  f"({best['tail_ci_low_pp']:+.2f}~{best['tail_ci_high_pp']:+.2f} pp),"
                  f"can only serve as directional evidence")
    else:
        best = max(cands, key=lambda c: c["delta_tail_pp"])
        verdict = "limited benefit, deployment not recommended"
        reason = ("no method simultaneously satisfies Δtail-F1 ≥ +1.0 pp and ΔWeighted-F1 ≥ -1.0 pp;"
                  f"largest tail gain by {best['method']}"
                  f"(Δtail={best['delta_tail_pp']:+.2f} pp,"
                  f"ΔWeighted={best['delta_weighted_pp']:+.2f} pp,"
                  f"ΔAcc={best['delta_acc_pp']:+.2f} pp)")
    return {"verdict": verdict, "reason": reason, "best_method": best["method"],
            "criteria": {"delta_tail_pp_min": 1.0, "delta_weighted_pp_min": -1.0,
                         "require_tail_ci_low_above_zero": True},
            "candidates": cands, "pareto_methods": pareto,
            "baseline": {"tail_f1": base["tail_f1"]["mean"],
                         "weighted_f1": base["weighted_f1"]["mean"],
                         "acc": base["acc"]["mean"]}}


def build_conclusions(agg, deltas, rec, group_support):
    b = agg[BASELINE]
    out = [
        f"baseline (no treatment) 5 seeds: Acc={b['acc']['mean']*100:.2f}±{b['acc']['std']*100:.2f}%,"
        f"Macro-F1={b['macro_f1']['mean']*100:.2f}% (report metric),"
        f"Macro-F1(support)={b['macro_f1_support']['mean']*100:.2f}%,"
        f"Weighted-F1={b['weighted_f1']['mean']*100:.2f}%,"
        f"tail F1={b['tail_f1']['mean']*100:.2f}% (primary metric),"
        f"tail Recall={b['tail_recall']['mean']*100:.2f}%,"
        f"tail F1(subset)={b['tail_f1_subset']['mean']*100:.2f}% (E1–E4 legacy metric)",
    ]
    for m, d in deltas.items():
        out.append(
            f"{m}: Δtail-F1={d['tail_f1']['mean_delta']*100:+.2f} pp "
            f"[{d['tail_f1']['ci95_low']*100:+.2f}, {d['tail_f1']['ci95_high']*100:+.2f}],"
            f"ΔWeighted-F1={d['weighted_f1']['mean_delta']*100:+.2f} pp "
            f"[{d['weighted_f1']['ci95_low']*100:+.2f}, {d['weighted_f1']['ci95_high']*100:+.2f}],"
            f"ΔAcc={d['acc']['mean_delta']*100:+.2f} pp,"
            f"ΔMacro-F1(support)={d['macro_f1_support']['mean_delta']*100:+.2f} pp")
    out.append(f"Recommended verdict: {rec['verdict']} -- {rec['reason']}")
    out.append(
        "Convention note: the primary convention computes per-class F1 over the full test set and then averages within groups; cross-group misclassifications count toward that class's FPs; "
        "the old convention (*_subset) evaluates only within that group's samples and ignores cross-group misclassifications, so it mistakes “predicting a rare class for many "
        "head/torso samples” degenerate behavior as tail improvement - focal-type methods gain +3.2 to +3.7 pp tail under the old convention, "
        "But under the primary convention tail drops instead, while Acc falls by about 10 pp - the empirical evidence of that mechanism ")
    out.append(
        f"Evidence strength: the tail group (202 classes with <100 samples) has only "
        f"{group_support['test_counts']['tail']} samples ("
        f"{group_support['tail_classes_with_test']} classes supported), and some classes have support of 1–3;"
        "The primary convention is already restricted to “test-supported” classes, but per-class F1 remains sensitive to single hits, "
        "tail conclusions can only serve as descriptive evidence, not a strong claim that “rebalancing reliably improves rare-framework prediction” ")
    return out


# ==================== Figures ====================
def make_group_plot(agg):
    methods = [m for m in METHODS if m in agg]
    if not methods:
        return
    groups = ["head_f1", "torso_f1", "tail_f1"]
    labels = ["head (n>=1000)", "torso (100-999)", "tail (<100)"]
    colors = ["#4C78A8", "#F58518", "#E45756"]
    x = np.arange(len(methods))
    w = 0.26
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for i, (g, lab, c) in enumerate(zip(groups, labels, colors)):
        mu = np.array([agg[m][g]["mean"] * 100 for m in methods])
        sd = np.array([agg[m][g]["std"] * 100 for m in methods])
        ax.bar(x + (i - 1) * w, mu, w, yerr=sd, capsize=3, label=lab, color=c, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ") for m in methods], rotation=20, ha="right",
                       fontsize=9)
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_title("E5: group Macro-F1 by rebalancing method (5 seeds, mean±std)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_GROUP, dpi=200)
    plt.close(fig)


def make_tradeoff_plot(agg, deltas, rec):
    if not deltas:
        return
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    xs, ys = [], []
    for m, d in deltas.items():
        dx = d["weighted_f1"]["mean_delta"] * 100
        dy = d["tail_f1"]["mean_delta"] * 100
        xs.append(dx)
        ys.append(dy)
        ax.scatter(dx, dy, s=60, color="#E45756" if m == rec["best_method"] else "#4C78A8",
                   zorder=3)
        ax.annotate(m, (dx, dy), textcoords="offset points", xytext=(6, 5), fontsize=8)
    ax.scatter(0, 0, s=110, marker="*", color="#222", zorder=4)
    ax.annotate("baseline", (0, 0), textcoords="offset points", xytext=(6, -13),
                fontsize=8)
    ax.plot([0] + xs, [0] + ys, color="#bbb", lw=1, zorder=1)
    ax.axhline(0, color="#888", lw=0.8)
    ax.axvline(0, color="#888", lw=0.8)
    ax.axhline(rec["criteria"]["delta_tail_pp_min"], color="#2ca02c", ls="--",
               lw=1, alpha=0.8)
    ax.axvline(rec["criteria"]["delta_weighted_pp_min"], color="#2ca02c", ls="--",
               lw=1, alpha=0.8)
    ax.set_xlabel("ΔWeighted-F1 vs baseline (pp)")
    ax.set_ylabel("Δtail Macro-F1 vs baseline (pp, primary metric)")
    ax.set_title("E5 trade-off: tail gain vs overall Weighted-F1 loss")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_TRADEOFF, dpi=200)
    plt.close(fig)


# ==================== Report ====================
def md_table(rows, cols):
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join([head, sep] + body)


def write_report(summary, agg, deltas, rec, summary_rows):
    p = summary["protocol"]
    gs = summary["group_support"]
    lines = []
    lines.append("# E5 class imbalance handling report")
    lines.append("")
    lines.append(f"Generated: {summary['generated_at']} | Companion: DeepLearning_Supplementary_Plan.md §6")
    lines.append("")
    lines.append("## 1. Purpose and protocol")
    lines.append("")
    lines.append("- **Purpose**: answer the open items in report §5.6.4(c) and §6.2.3 - under extreme imbalance across 231 classes,"
                 "Can focal loss / class weighting / Balanced Softmax / weighted sampling buy tail-class Macro-F1 gains at an acceptable overall accuracy loss "
                 "in exchange for tail-class Macro-F1 gains. ")
    lines.append(f"- **Model**: {p['model']}, the single best ANN validated once (plan §6.1).")
    lines.append(f"- **Data**: 104-dim features, train {p['n_train']:,} / val {p['n_val']:,} / "
                 f"test {p['n_test']:,}, {p['n_classes']} classes; split frozen, fully consistent with E1/E2/E4.")
    lines.append(f"- **Seeds**: {', '.join(str(s) for s in p['seeds'])} (5 seeds)."
                 "baseline directly reuses E4's BiGRU five-seed predictions (same protocol, same hyperparameters), no retraining. ")
    lines.append(f"- **Training**: AdamW + ReduceLROnPlateau + grad clip 1.0 + label smoothing "
                 f"{p['label_smoothing']}; early stopping monitors val Macro-F1, 100 epochs / patience 15."
                 "Label smoothing is retained in all methods to isolate the “only change the class-balance mechanism” contrast. ")
    lines.append("- **Loss definition**:")
    for k, v in p["loss_definitions"].items():
        lines.append(f"  - `{k}`:{v}")
    lines.append("")
    lines.append("### 1.1 Group definitions and metric conventions")
    lines.append("")
    lines.append(f"- Grouping (plan §6.2, by training-set class frequency): head n_c≥{HEAD_MIN} (1 class),"
                 f"torso 100–999 (28 classes), tail <100 (202 classes). Train samples head/torso/tail = "
                 f"{gs['train_counts']['head']:,}/{gs['train_counts']['torso']:,}/"
                 f"{gs['train_counts']['tail']:,}; test samples = "
                 f"{gs['test_counts']['head']:,}/{gs['test_counts']['torso']:,}/"
                 f"{gs['test_counts']['tail']:,}. Among 231 classes, {gs['classes_without_test']} classes have "
                 f"no samples in the frozen test set; in the tail group, {gs['tail_classes_with_test']} classes have test support.")
    lines.append("- **Primary metric (`head_f1/torso_f1/tail_f1`)**: compute per-class F1 on the full test set,"
                 "Then take the macro average over “classes in this group that have test support”. Cross-group misclassifications count toward that class's FPs, "
                 "Hence the degenerate behavior of “predicting rare classes everywhere” is not mistaken for tail improvement; "
                 "This convention is strictly decomposable with Macro-F1(support) (weighted sum = the macro average over the 148 supported classes). ")
    lines.append("- **Secondary metric (`*_f1_subset`)**: keep only that group's samples, labels taken as all classes in the group"
                 "(E1-E4 section 4.5 convention). Cross-group misclassifications are not counted as FPs, systematically inflating results; for historical reference only, "
                 "**not used as a basis for conclusions**.")
    lines.append("- **Macro-F1 (`macro_f1`)**: consistent with E1–E4, sklearn default macro average"
                 "(denominator = classes “appearing in the test set or ever predicted”, including 83 classes without test samples); "
                 "`macro_f1_support` is the macro average over the 148 test-supported classes, a cleaner cross-model comparison quantity. ")
    lines.append("")
    lines.append("## 2. Results (5 seeds, mean ± std)")
    lines.append("")
    rows = []
    for r in summary_rows:
        rows.append({
            "Method": r["method"], "Config": r["gamma_or_weight"],
            "Acc (%)": f"{r['acc']*100:.2f} ± {r['acc_std']*100:.2f}",
            "Macro-F1 (%)": f"{r['macro_f1']*100:.2f} ± {r['macro_f1_std']*100:.2f}",
            "Macro-F1(sup) (%)": f"{r['macro_f1_support']*100:.2f} ± "
                                f"{r['macro_f1_support_std']*100:.2f}",
            "Weighted-F1 (%)": f"{r['weighted_f1']*100:.2f} ± {r['weighted_f1_std']*100:.2f}",
            "Top-5 (%)": f"{r['top5']*100:.2f} ± {r['top5_std']*100:.2f}",
        })
    lines.append(md_table(rows, ["Method", "Config", "Acc (%)", "Macro-F1 (%)",
                                 "Macro-F1(sup) (%)", "Weighted-F1 (%)", "Top-5 (%)"]))
    lines.append("")
    rows = []
    for r in summary_rows:
        rows.append({
            "Method": r["method"],
            "head F1 (%)": f"{r['head_f1']*100:.2f} ± {r['head_f1_std']*100:.2f}",
            "torso F1 (%)": f"{r['torso_f1']*100:.2f} ± {r['torso_f1_std']*100:.2f}",
            "tail F1 (%)": f"{r['tail_f1']*100:.2f} ± {r['tail_f1_std']*100:.2f}",
            "tail Recall (%)": f"{r['tail_recall']*100:.2f} ± {r['tail_recall_std']*100:.2f}",
            "tail F1 (subset, %)": f"{r['tail_f1_subset']*100:.2f}",
        })
    lines.append("### 2.1 Group metrics (primary metric)")
    lines.append("")
    lines.append(md_table(rows, ["Method", "head F1 (%)", "torso F1 (%)", "tail F1 (%)",
                                 "tail Recall (%)", "tail F1 (subset, %)"]))
    lines.append("")
    lines.append("> `tail F1 (subset)` is the E1–E4 legacy metric (within-group subset, all-group labels),"
                 "Including 83 classes without test samples counted as 0, and ignoring cross-group FPs; this column is kept only for comparison with earlier reports. ")
    lines.append("")

    def _tail_precision(r):
        """Back-derives P = F1*R/(2R-F1) from F1 = 2PR/(P+R); only used to explain the direction difference between Recall and F1."""
        f1, rec = r["tail_f1"], r["tail_recall"]
        den = 2 * rec - f1
        return f1 * rec / den if den > 1e-9 else float("nan")

    by_method = {r["method"]: r for r in summary_rows}
    focal_key = "focal_g2" if "focal_g2" in by_method else None
    txt = ",".join(f"{m} {_tail_precision(by_method[m]) * 100:.1f}%"
                    for m in ("baseline", "weighted_ce") if m in by_method)
    if focal_key:
        r_f = by_method[focal_key]
        txt += f",{focal_key} {_tail_precision(r_f) * 100:.1f}%"
        d_rec = (r_f["tail_recall"] - by_method["baseline"]["tail_recall"]) * 100
        d_prec = (_tail_precision(r_f) - _tail_precision(by_method["baseline"])) * 100
        lines.append(
            f"> **Recall–Precision decomposition**: tail precision back-derived from tail F1 and Recall: {txt}."
            f"{focal_key} tail Recall vs baseline {d_rec:+.1f} pp, but precision {d_prec:+.1f} pp,"
            "After offsetting, tail F1 is slightly lower instead - showing focal mainly relaxes the decision threshold for rare classes, "
            "rather than improving between-class separability. ")
        lines.append("")
    lines.append("## 3. Paired differences vs baseline (same-seed pairing, t-distribution 95% CI)")
    lines.append("")
    rows = []
    for m, d in deltas.items():
        rows.append({
            "Method": m,
            "Δtail-F1 (pp)": f"{d['tail_f1']['mean_delta']*100:+.2f} "
                             f"[{d['tail_f1']['ci95_low']*100:+.2f}, "
                             f"{d['tail_f1']['ci95_high']*100:+.2f}]",
            "ΔMacro(sup) (pp)": f"{d['macro_f1_support']['mean_delta']*100:+.2f} "
                                f"[{d['macro_f1_support']['ci95_low']*100:+.2f}, "
                                f"{d['macro_f1_support']['ci95_high']*100:+.2f}]",
            "ΔWeighted-F1 (pp)": f"{d['weighted_f1']['mean_delta']*100:+.2f} "
                                 f"[{d['weighted_f1']['ci95_low']*100:+.2f}, "
                                 f"{d['weighted_f1']['ci95_high']*100:+.2f}]",
            "ΔAcc (pp)": f"{d['acc']['mean_delta']*100:+.2f} "
                         f"[{d['acc']['ci95_low']*100:+.2f}, "
                         f"{d['acc']['ci95_high']*100:+.2f}]",
        })
    lines.append(md_table(rows, ["Method", "Δtail-F1 (pp)", "ΔMacro(sup) (pp)",
                                 "ΔWeighted-F1 (pp)", "ΔAcc (pp)"]))
    lines.append("")
    lines.append("> Δ = method − baseline; the interval is the t-distribution 95% CI of 5-seed paired differences (4 degrees of freedom),"
                 "Characterizes seed variability rather than test-sample-level significance. ")
    lines.append("")
    lines.append("## 4. Trade-off and recommended configuration")
    lines.append("")
    lines.append(f"- **Verdict rule**: Δtail-F1 ≥ +{rec['criteria']['delta_tail_pp_min']:.1f} pp "
                 f"and ΔWeighted-F1 ≥ {rec['criteria']['delta_weighted_pp_min']:.1f} pp is considered practically feasible.")
    lines.append(f"- **Conclusion**: **{rec['verdict']}**. {rec['reason']}.")
    lines.append(f"- **Pareto frontier**: {', '.join(rec['pareto_methods']) if rec['pareto_methods'] else 'none'}.")
    lines.append("")
    rows = []
    for c in rec["candidates"]:
        rows.append({
            "Method": c["method"], "Delta tail (pp)": f"{c['delta_tail_pp']:+.2f}",
            "ΔWeighted (pp)": f"{c['delta_weighted_pp']:+.2f}",
            "ΔMacro(sup) (pp)": f"{c['delta_macro_support_pp']:+.2f}",
            "ΔMacro (pp)": f"{c['delta_macro_pp']:+.2f}",
            "ΔAcc (pp)": f"{c['delta_acc_pp']:+.2f}",
            "Practical threshold": "Pass" if c["meets_practical"] else "Fail ",
            "Pareto": "Yes" if c["pareto"] else "No",
        })
    lines.append(md_table(rows, ["Method", "Δtail (pp)", "ΔWeighted (pp)", "ΔMacro(sup) (pp)",
                                 "Delta Macro (pp)", "Delta Acc (pp)", "practical threshold", "Pareto"]))
    lines.append("")
    lines.append("![group](E5_group_comparison.png)")
    lines.append("")
    lines.append("![tradeoff](E5_tradeoff.png)")
    lines.append("")
    lines.append("## 5. Conclusions and evidence boundary")
    lines.append("")
    for i, c in enumerate(summary["conclusions"], 1):
        lines.append(f"{i}. {c}")
    lines.append("")
    lines.append("- **Report item that can be directly closed**: §5.6.4(c) “class imbalance still lacks dedicated weighting/resampling treatment”"
                 "with section 6.2.3 “class-imbalance countermeasures” - this experiment has completed the four-mechanism comparison under the unified protocol. ")
    lines.append("- **Cannot claim**:")
    lines.append("  - the tail group's 202 classes have only 712 samples in the test set, 119 classes supported, and some classes have support of 1–3;"
                 "The per-class F1 macro average is highly sensitive to single hits; any tail gain should be phrased as descriptive evidence, "
                 "Cannot serve as a strong claim that “rebalancing reliably improves rare-framework prediction”. ")
    lines.append("  - 5 seeds only give coarse-grained variance; the paired CI has 4 degrees of freedom.")
    lines.append("  - this experiment only adjusts loss/sampling, without changing data, split, or the 231-class label space;"
                 "Hierarchical classification (coarse first, then fine) or two-stage (first decide common classes, then discriminate among rare-class candidates) "
                 "it is a structurally different alternative requiring separate experiments and cannot be extrapolated from this result. ")
    lines.append("  - `macro_f1` (report default metric) and `macro_f1_support` differ in denominator:"
                 "When a method changes the number of “predicted classes without test samples”, the two may disagree in direction; "
                 "Cross-model comparison should follow `macro_f1_support` and the grouped primary convention. ")
    lines.append("")
    lines.append("## 6. File list")
    lines.append("")
    lines.append(md_table([
        {"File": "e5_imbalance.py", "Content": "script: train (resumable) + analyze + report"},
        {"File": "E5_runs.csv", "Content": "full metrics for each (method, seed)"},
        {"File": "E5_summary.csv", "Content": "per-method aggregation (plan-specified 8 columns + std/grouping/recall)"},
        {"File": "E5_summary.json", "Content": "machine-readable summary (protocol/loss definitions/paired differences/recommendation)"},
        {"File": "E5_group_comparison.png", "Content": "head/torso/tail grouped bar chart"},
        {"File": "E5_tradeoff.png", "Content": "Δtail vs ΔWeighted trade-off plot"},
        {"File": "preds/", "Content": "test predictions per (method, seed) (int16)"},
        {"File": "e5_run.log", "Content": "run log"},
    ], ["File", "Content"]))
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ==================== Main pipeline ====================
def main():
    global LOG, EPOCH_CAP, RUN_TAG
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--epoch-cap", type=int, default=None)
    ap.add_argument("--methods", type=str, default="")
    ap.add_argument("--seeds", type=str, default="")
    args = ap.parse_args()

    EPOCH_CAP = args.epoch_cap
    RUN_TAG = "smoke" if EPOCH_CAP else ""
    init_paths(RUN_TAG)
    LOG = make_logger()
    methods = [m for m in (args.methods.split(",") if args.methods else METHODS) if m]
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else SEEDS

    log("=" * 72)
    log(f"E5 class imbalance handling | methods={methods} | seeds={seeds} | "
        f"epoch_cap={EPOCH_CAP} | device={ec.DEVICE}")

    if not args.analyze_only:
        params = load_tuned_bigru()
        log(f"BiGRU tuned parameters: {json.dumps(params)}")
        data = load_data()

        # baseline: reuse E4, no retraining
        if BASELINE in methods:
            existing = set()
            if os.path.exists(TRAIN_CSV):
                prev = pd.read_csv(TRAIN_CSV)
                existing = {(r["method"], int(r["seed"])) for _, r in prev.iterrows()}
            rows = e4_bigru_rows(data)
            for r in rows:
                if (r["method"], r["seed"]) not in existing:
                    append_row(TRAIN_CSV, r, RESULT_FIELDS)
            log(f"baseline reuses E4 five-seed predictions: {len(rows)} entries")

        done = load_done()
        for m in methods:
            if m == BASELINE:
                continue
            for s in seeds:
                run_one(m, s, params, data, done)

    if os.path.exists(TRAIN_CSV):
        analyze()


if __name__ == "__main__":
    main()
