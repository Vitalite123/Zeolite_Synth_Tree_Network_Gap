"""
E7 model capacity sweep
=====================================================================================
Companion documents: CBUTransfer/DeepLearning/DeepLearning_Supplementary_Plan.md §8 (E7)

Design:
  2 architectures x 3 capacity tiers x 5 seeds = 30 training runs.
  BiGRU   : capacity tuned via hidden_size / num_layers (sequence layout fixed 26x4, consistent with E1.2/E4/E6 main config);
  FT-Trans: capacity tuned via d_token / n_blocks (other hyperparameters frozen to E2.1 tuned values).
  Three capacity targets: Small ~0.1-0.3M / Medium ~1-3M (current scale) / Large ~8-15M.

Training protocol:
  BiGRU   : AdamW + ReduceLROnPlateau(factor .5, patience 5) + grad clip 1.0
            + label smoothing 0.1, 100 epoch / patience 15 (same as E1.2/E4/E5/E6).
  FT-Trans: AdamW, no scheduler, no grad clip, no label smoothing,
            100 epochs / patience 15 (same protocol as E4's FT-Transformer, ensuring comparability at the medium fraction).
  early stopping monitors val Macro-F1; seeds {42,123,777,2026,3407}.

Metrics:
  test: Accuracy / Macro-F1 / Macro-F1(sup) / Weighted-F1 / Top-5
        + head/torso/tail (E5 main convention: per-class F1 over the full test set -> macro average over classes supported within the group)
  Capacity: train_macro_f1 / val_macro_f1 / gap (train-val), all computed as mean±std per seed.

Verdict (plan §8.2, pre-defined, both outcomes reported):
  Paired differences Δval and Δgap for Large vs Medium:
    Δval <= +0.5 pp and Δgap >= +2.0 pp  -> "sufficient capacity, data-limited"
    Δval >= +1.0 pp and CI lower bound > 0      -> "capacity bottleneck (keeps improving)"
    Δval >= +1.0 pp but CI crosses 0          -> "suspected capacity bottleneck (directional, insufficient evidence)"
    everything else                                -> "uncertain (intermediate)"

Usage:
  py -B e7_capacity.py                     # train + analyze
  py -B e7_capacity.py --analyze-only      # recompute from written predictions and generate report only
  py -B e7_capacity.py --epoch-cap 1       # smoke (artifacts get _smoke suffix)
  py -B e7_capacity.py --archs BiGRU --capacities small --seeds 42
"""

import argparse
import csv
import gc
import inspect
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
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import t as tdist
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(os.path.abspath(__file__))
SUP_DIR = os.path.abspath(os.path.join(HERE, ".."))
E1_DIR = os.path.join(SUP_DIR, "E1_tuning")
E2_DIR = os.path.join(SUP_DIR, "E2_sota")
for _p in (E1_DIR, E2_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import rtdl  # noqa: E402
import e2_common as ec  # noqa: E402

# ==================== Constants ====================
SEEDS = [42, 123, 777, 2026, 3407]
N_CLASSES = ec.N_CLASS_EXPECT
MAX_EPOCHS = 100
PATIENCE = 15
HEAD_MIN, TORSO_MIN = 1000, 100

SEQ_LEN, FEAT_PER_STEP = 26, 4          # consistent with the E1.2/E4/E6 main config
CAPACITIES = ["small", "medium", "large"]

# Capacity tiers (parameter counts measured during smoke tests; measured values are used in the report)
CONFIGS = {
    "BiGRU": {
        "small": dict(hidden_size=64, num_layers=1),
        "medium": dict(hidden_size=256, num_layers=1),
        "large": dict(hidden_size=512, num_layers=2),
    },
    "FTT": {
        "small": dict(d_token=64, n_blocks=2, n_heads=8),
        "medium": dict(d_token=96, n_blocks=3, n_heads=4),
        "large": dict(d_token=416, n_blocks=4, n_heads=8),
    },
}

RESULT_FIELDS = [
    "arch", "capacity", "seed", "n_params", "batch_used", "best_epoch", "n_epochs",
    "duration_s",
    "train_acc", "train_macro_f1", "val_acc", "val_macro_f1",
    "gap_acc", "gap_macro_f1",
    "acc", "macro_f1", "macro_f1_support", "weighted_f1", "top5",
    "head_f1", "torso_f1", "tail_f1",
    "hidden_size", "num_layers", "d_token", "n_blocks", "n_heads",
]
AGG_COLS = ["train_acc", "train_macro_f1", "val_acc", "val_macro_f1",
            "gap_acc", "gap_macro_f1",
            "acc", "macro_f1", "macro_f1_support", "weighted_f1", "top5",
            "head_f1", "torso_f1", "tail_f1"]

RUN_CSV = SCAN_CSV = SUMMARY_JSON = REPORT_MD = FIG = PRED_DIR = LOG_PATH = None
LOG = None
EPOCH_CAP = None
RUN_TAG = ""


def init_paths(tag=""):
    global RUN_CSV, SCAN_CSV, SUMMARY_JSON, REPORT_MD, FIG, PRED_DIR, LOG_PATH
    stamp = f"_{tag}" if tag else ""
    RUN_CSV = os.path.join(HERE, f"E7_runs{stamp}.csv")
    SCAN_CSV = os.path.join(HERE, f"capacity_scan{stamp}.csv")
    SUMMARY_JSON = os.path.join(HERE, f"E7_summary{stamp}.json")
    REPORT_MD = os.path.join(HERE, f"E7_report{stamp}.md")
    FIG = os.path.join(HERE, f"capacity_curve{stamp}.png")
    PRED_DIR = os.path.join(HERE, f"preds{stamp}")
    LOG_PATH = os.path.join(HERE, f"e7_run{stamp}.log")
    os.makedirs(PRED_DIR, exist_ok=True)


def log(msg):
    if LOG is not None:
        LOG.info(msg)
    else:
        print(msg, flush=True)


def make_logger():
    import logging
    lg = logging.getLogger("E7")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
        for h in (logging.FileHandler(LOG_PATH, encoding="utf-8"),
                  logging.StreamHandler()):
            h.setFormatter(fmt)
            lg.addHandler(h)
    return lg


# ==================== Model: BiGRU (same generic implementation as E6) ====================
class BiGRUSeq(nn.Module):
    """Generic version of E1.2 BiGRU; at (26, 4) it is layer-wise equivalent to e1_2_bigru.BiGRU_Zeolite."""

    def __init__(self, num_classes, hidden_size, num_layers, dropout,
                 seq_len, feature_per_step):
        super().__init__()
        self.seq_len = int(seq_len)
        self.feature_per_step = int(feature_per_step)
        self.feature_embedding = nn.Linear(self.feature_per_step, hidden_size // 2)
        self.gru = nn.GRU(
            input_size=hidden_size // 2, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1))
        self.fc1 = nn.Linear(hidden_size * 2, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), self.seq_len, self.feature_per_step)
        x = self.feature_embedding(x)
        gru_out, _ = self.gru(x)
        attn = torch.softmax(self.attention(gru_out), dim=1)
        context = torch.sum(attn * gru_out, dim=1)
        out = self.dropout1(self.relu(self.bn1(self.fc1(context))))
        out = self.dropout2(self.relu(self.bn2(self.fc2(out))))
        return self.fc3(out)


def forward_bigru(model, xb):
    return model(xb)


# ==================== Model: FT-Transformer (equivalent to E2.1 build_ftt) ====================
BASELINE_SIG = (set(inspect.signature(rtdl.FTTransformer.make_baseline).parameters)
                if hasattr(rtdl.FTTransformer, "make_baseline") else None)


def build_ftt(params, n_classes):
    """Built against the actual rtdl 0.0.13 signature; prenormalization=True.
    equivalent to E2_sota/e2_1_ft_transformer.build_ftt (copied here so that importing does not
    create a log handler in the E2 directory)."""
    if BASELINE_SIG:
        kwargs = dict(
            n_num_features=104, cat_cardinalities=None, d_out=n_classes,
            d_token=params["d_token"], n_blocks=params["n_blocks"],
            attention_n_heads=params["n_heads"],
            ffn_d_hidden=int(params["d_token"] * params["ffn_mult"]),
            ffn_dropout=params["ffn_dropout"],
            residual_dropout=params["resid_dropout"],
            prenormalization=True,
        )
        for k in ("attention_dropout", "attention_probs_dropout"):
            if k in BASELINE_SIG:
                kwargs[k] = params["attn_dropout"]
                break
        kwargs = {k: v for k, v in kwargs.items() if k in BASELINE_SIG}
        return rtdl.FTTransformer.make_baseline(**kwargs)
    return rtdl.FTTransformer.make_default(
        n_num_features=104, cat_cardinalities=None, d_out=n_classes,
        n_blocks=params["n_blocks"])


def forward_ftt(model, xb):
    return model(xb, None)


# ==================== Hyperparameters (frozen) ====================
def load_bigru_tuned():
    with open(os.path.join(E1_DIR, "best_params_E1_2.json"), "r", encoding="utf-8") as f:
        return json.load(f)["params"]


def build_specs(ftt_batch=None):
    """Returns {arch: {capacity: spec}}; spec contains build/forward/params/protocol fields."""
    bg = load_bigru_tuned()
    ftt = {
        "n_heads": 4, "attn_dropout": 0.11587410348784977, "ffn_mult": 4.0,
        "ffn_dropout": 0.15033196742261523, "resid_dropout": 0.04999910597040305,
        "lr": 0.0006950868415415626, "weight_decay": 6.97268811461115e-05,
        # E2.1/E4 use batch 512, but on this machine's 2GB MX450, 512 reliably OOMs in testing;
        # All three tiers use 256 as the base; the Large tier then drops to 64 via OOM retry (actual value recorded in batch_used)
        "batch_size": 256,
    }
    if ftt_batch:
        ftt["batch_size"] = int(ftt_batch)
    specs = {"BiGRU": {}, "FTT": {}}
    for cap, cfg in CONFIGS["BiGRU"].items():
        params = dict(
            hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"],
            dropout_rate=float(bg["dropout_rate"]),
            learning_rate=float(bg["learning_rate"]),
            weight_decay=float(bg["weight_decay"]),
            batch_size=int(bg["batch_size"]),
        )
        specs["BiGRU"][cap] = dict(
            params=params,
            build=lambda p: BiGRUSeq(N_CLASSES, int(p["hidden_size"]),
                                     int(p["num_layers"]), float(p["dropout_rate"]),
                                     SEQ_LEN, FEAT_PER_STEP),
            forward=forward_bigru,
            scheduler="plateau", label_smoothing=0.1, grad_clip=1.0,
            cap_kwargs=cfg,
        )
    for cap, cfg in CONFIGS["FTT"].items():
        params = dict(ftt)
        params.update(d_token=cfg["d_token"], n_blocks=cfg["n_blocks"],
                      n_heads=cfg["n_heads"])
        specs["FTT"][cap] = dict(
            params=params,
            build=lambda p: build_ftt(p, N_CLASSES),
            forward=forward_ftt,
            scheduler=None, label_smoothing=0.0, grad_clip=None,
            cap_kwargs=cfg,
        )
    return specs


# ==================== Grouping and evaluation (primary convention, same as E5/E6) ====================
def make_threshold_strata(y_train, n_classes):
    counts = np.bincount(y_train, minlength=n_classes)
    return {
        "head": np.where(counts >= HEAD_MIN)[0],
        "torso": np.where((counts >= TORSO_MIN) & (counts < HEAD_MIN))[0],
        "tail": np.where(counts < TORSO_MIN)[0],
    }


def evaluate_pred(pred, y, data):
    labels = np.arange(N_CLASSES)
    supported = np.unique(y)
    f1_all = f1_score(y, pred, average=None, labels=labels, zero_division=0)
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
    return out


# ==================== Data ====================
def load_data():
    data = ec.load_data(LOG)
    data["thr_strata"] = make_threshold_strata(data["y_train"], N_CLASSES)
    for key in ("head", "torso", "tail"):
        idx = data["thr_strata"][key]
        log(f"group ({key}): {len(idx)} classes | "
            f"test {int(np.isin(data['y_test'], idx).sum())}")
    return data


# ==================== Training ====================
def _train_once(spec, seed, data, bs, tag):
    ec.set_seed(seed)
    p = spec["params"]
    build, forward = spec["build"], spec["forward"]
    Xtr, ytr = data["X_train"], data["y_train"]
    Xval, yval = data["X_val"], data["y_val"]

    ds = TensorDataset(torch.as_tensor(Xtr), torch.as_tensor(ytr, dtype=torch.long))
    loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0,
                        pin_memory=(ec.DEVICE.type == "cuda"))

    model = build(p).to(ec.DEVICE)
    n_params = sum(x.numel() for x in model.parameters())
    lr = float(p.get("lr", p.get("learning_rate")))
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=float(p.get("weight_decay", 0.0)))
    sched = (torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min",
                                                        factor=0.5, patience=5)
             if spec["scheduler"] == "plateau" else None)
    ls = float(spec["label_smoothing"])
    clip = spec["grad_clip"]
    max_epochs = min(MAX_EPOCHS, EPOCH_CAP) if EPOCH_CAP else MAX_EPOCHS
    patience = min(PATIENCE, max(1, max_epochs // 2))

    best_f1, best_state, best_epoch, bad = -1.0, None, 0, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        total, n_batches = 0.0, 0
        for xb, yb in loader:
            if xb.size(0) == 1:
                continue
            xb = xb.to(ec.DEVICE, non_blocking=True)
            yb = yb.to(ec.DEVICE, non_blocking=True)
            opt.zero_grad()
            loss = F.cross_entropy(forward(model, xb), yb, label_smoothing=ls)
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
            opt.step()
            total += float(loss.item())
            n_batches += 1

        t_eval = time.time()
        val_proba = ec.predict_proba(model, Xval, forward)
        val_f1 = ec.metrics_from_proba(val_proba, yval, N_CLASSES)["macro_f1"]
        p_true = val_proba[np.arange(len(yval)), yval]
        val_loss = float(-np.log(np.clip(p_true, 1e-12, 1.0)).mean())
        if sched is not None:
            sched.step(val_loss)

        improved = val_f1 > best_f1
        log(f"{tag} epoch {epoch:3d} | train_loss={total/max(n_batches,1):.4f} | "
            f"val_loss={val_loss:.4f} | val_MacroF1={val_f1:.4f} | "
            f"eval={time.time()-t_eval:.1f}s{' *' if improved else ''}")
        if improved:
            best_f1, best_epoch, bad = val_f1, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    train_proba = ec.predict_proba(model, Xtr, forward)
    val_proba = ec.predict_proba(model, Xval, forward)
    test_proba = ec.predict_proba(model, data["X_test"], forward)
    train_m = ec.metrics_from_proba(train_proba, ytr, N_CLASSES)
    val_m = ec.metrics_from_proba(val_proba, yval, N_CLASSES)
    test_m = evaluate_pred(test_proba.argmax(axis=1), data["y_test"], data)
    top5 = ec.metrics_from_proba(test_proba, data["y_test"], N_CLASSES)["top5"]
    n_epochs = epoch
    log(f"{tag} best_epoch={best_epoch}/{n_epochs} | "
        f"train MacroF1={train_m['macro_f1']:.4f} val MacroF1={best_f1:.4f} | "
        f"test Acc={test_m['acc']:.4f} MacroF1={test_m['macro_f1']:.4f} | "
        f"params={n_params} batch={bs}")
    del model
    if ec.DEVICE.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
    return dict(test_proba=test_proba,
                train_acc=train_m["acc"], train_macro_f1=train_m["macro_f1"],
                val_acc=val_m["acc"], val_macro_f1=val_m["macro_f1"],
                gap_acc=float(train_m["acc"] - val_m["acc"]),
                gap_macro_f1=float(train_m["macro_f1"] - val_m["macro_f1"]),
                test_m=test_m, top5=top5,
                n_params=int(n_params), best_epoch=int(best_epoch),
                n_epochs=int(n_epochs), batch_used=int(bs))


def train_with_oom_retry(spec, seed, data, tag):
    bs = int(spec["params"]["batch_size"])
    if ec.DEVICE.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
    # FTT Large (d_token=416/4 layers) must drop to 64 under 2GB VRAM; hence the retry floor is relaxed to 32
    for attempt in range(5):
        try:
            return _train_once(spec, seed, data, bs, tag)
        except torch.OutOfMemoryError:
            if attempt == 4 or bs // 2 < 32:
                raise
            bs = max(32, bs // 2)
            log(f"{tag} | CUDA OOM, batch -> {bs}")
            gc.collect()
            torch.cuda.empty_cache()


# ==================== CSV / resumable runs ====================
def append_row(path, row, fields):
    new = not os.path.exists(path)
    mode = "w" if new else "a"
    enc = "utf-8-sig" if new else "utf-8"
    with open(path, mode, newline="", encoding=enc) as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)


def pred_path(arch, cap, seed):
    return os.path.join(PRED_DIR, f"{arch}_{cap}_seed{seed}.npy")


def load_done():
    if not os.path.exists(RUN_CSV):
        return set()
    df = pd.read_csv(RUN_CSV)
    return {(r["arch"], r["capacity"], int(r["seed"])) for _, r in df.iterrows()
            if os.path.exists(pred_path(r["arch"], r["capacity"], int(r["seed"])))}


# ==================== Single run ====================
def run_one(arch, cap, spec, seed, data, done):
    if (arch, cap, seed) in done:
        log(f"[skip] {arch}/{cap} seed={seed} already done")
        return
    tag = f"[{arch}/{cap} seed={seed}]"
    log(f"=== {tag} start ===")
    t0 = time.time()
    out = train_with_oom_retry(spec, seed, data, tag)
    m = out["test_m"]
    np.save(pred_path(arch, cap, seed),
            out["test_proba"].argmax(axis=1).astype(np.int16))
    kw = spec["cap_kwargs"]
    row = {
        "arch": arch, "capacity": cap, "seed": int(seed),
        "n_params": int(out["n_params"]), "batch_used": int(out["batch_used"]),
        "best_epoch": int(out["best_epoch"]), "n_epochs": int(out["n_epochs"]),
        "duration_s": round(time.time() - t0, 1),
        "train_acc": round(out["train_acc"], 6),
        "train_macro_f1": round(out["train_macro_f1"], 6),
        "val_acc": round(out["val_acc"], 6),
        "val_macro_f1": round(out["val_macro_f1"], 6),
        "gap_acc": round(out["gap_acc"], 6),
        "gap_macro_f1": round(out["gap_macro_f1"], 6),
        "top5": round(float(out["top5"]), 6),
        "hidden_size": kw.get("hidden_size", ""),
        "num_layers": kw.get("num_layers", ""),
        "d_token": kw.get("d_token", ""),
        "n_blocks": kw.get("n_blocks", ""),
        "n_heads": kw.get("n_heads", ""),
    }
    for k in ("acc", "macro_f1", "macro_f1_support", "weighted_f1",
              "head_f1", "torso_f1", "tail_f1"):
        row[k] = round(float(m[k]), 6)
    append_row(RUN_CSV, row, RESULT_FIELDS)
    log(f"=== {tag} done in {row['duration_s']}s | Acc={row['acc']:.4f} "
        f"MacroF1={row['macro_f1']:.4f} | val={row['val_macro_f1']:.4f} "
        f"gap={row['gap_macro_f1']:.4f} ===")


# ==================== Statistics ====================
def mean_std(v):
    v = np.asarray(v, dtype=float)
    if len(v) == 0:
        return np.nan, np.nan
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0)


def paired_delta(a_by_seed, b_by_seed):
    common = sorted(set(a_by_seed) & set(b_by_seed))
    if not common:
        return None
    d = np.array([a_by_seed[s] - b_by_seed[s] for s in common], dtype=float)
    n = len(d)
    mu = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    half = float(tdist.ppf(0.975, n - 1) * sd / math.sqrt(n)) if n > 1 else float("nan")
    return {"n_seeds": n, "mean_delta": mu, "ci95_low": mu - half,
            "ci95_high": mu + half, "seeds": [int(s) for s in common],
            "per_seed": {int(s): float(x) for s, x in zip(common, d)}}


def series(df, arch, cap, col):
    sub = df[(df["arch"] == arch) & (df["capacity"] == cap)]
    return {int(r["seed"]): float(r[col]) for _, r in sub.iterrows()}


# ==================== Verdict ====================
STALL_PP, GAP_PP, IMPROVE_PP = 0.005, 0.02, 0.01


def judge(d_val, d_gap):
    if d_val is None or d_gap is None:
        return "insufficient data"
    dv = d_val["mean_delta"]
    dg = d_gap["mean_delta"]
    if dv <= STALL_PP and dg >= GAP_PP:
        return "sufficient capacity, data-limited (val stagnates while the train/val gap widens)"
    if dv >= IMPROVE_PP and d_val["ci95_low"] > 0:
        return "capacity bottleneck (Large still improves significantly, underfitting conclusion needs revision)"
    if dv >= IMPROVE_PP:
        return "suspected capacity bottleneck (improves but 95% CI crosses 0, directional evidence)"
    return "uncertain (intermediate, no criterion reached)"


# ==================== Analysis ====================
def analyze():
    log("=== Analysis phase ===")
    if not os.path.exists(RUN_CSV):
        raise RuntimeError(f"missing run log: {RUN_CSV}")
    runs = pd.read_csv(RUN_CSV)
    runs = (runs.drop_duplicates(subset=["arch", "capacity", "seed"], keep="last")
                .sort_values(["arch", "capacity", "seed"]).reset_index(drop=True))
    runs.to_csv(RUN_CSV, index=False, encoding="utf-8-sig")

    rows = []
    for arch in ("BiGRU", "FTT"):
        for cap in CAPACITIES:
            sub = runs[(runs["arch"] == arch) & (runs["capacity"] == cap)]
            if sub.empty:
                log(f"[warn] missing {arch}/{cap}, skipping")
                continue
            row = {"arch": arch, "capacity": cap, "n_seeds": int(sub["seed"].nunique()),
                   "n_params": int(sub["n_params"].iloc[0]),
                   "batch_used": int(sub["batch_used"].iloc[0])}
            for col in AGG_COLS:
                mu, sd = mean_std(sub[col].values)
                row[col] = round(mu, 6)
                row[f"{col}_std"] = round(sd, 6)
            for col in ("best_epoch", "n_epochs", "duration_s"):
                mu, sd = mean_std(sub[col].values)
                row[f"{col}_mean"] = round(mu, 2)
                row[f"{col}_std"] = round(sd, 2)
            rows.append(row)
    scan_df = pd.DataFrame(rows)
    scan_df.to_csv(SCAN_CSV, index=False, encoding="utf-8-sig")

    # ---------- Paired differences ----------
    paired = {}
    for arch in ("BiGRU", "FTT"):
        if not set(CAPACITIES) <= set(scan_df[scan_df["arch"] == arch]["capacity"]):
            continue
        peri = {}
        for higher, lower in (("medium", "small"), ("large", "medium"),
                             ("large", "small")):
            peri[f"{higher}_minus_{lower}"] = {
                col: paired_delta(series(runs, arch, higher, col),
                                  series(runs, arch, lower, col))
                for col in ("val_macro_f1", "val_acc", "train_macro_f1",
                            "gap_macro_f1", "gap_acc", "acc", "macro_f1")
            }
        paired[arch] = peri

    verdicts = {}
    for arch, peri in paired.items():
        d_val = peri["large_minus_medium"]["val_macro_f1"]
        d_gap = peri["large_minus_medium"]["gap_macro_f1"]
        verdicts[arch] = {
            "verdict": judge(d_val, d_gap),
            "d_val_macro_f1_large_minus_medium": d_val,
            "d_gap_macro_f1_large_minus_medium": d_gap,
        }

    summary = {
        "experiment": "E7_capacity",
        "date": "2026-10-25",
        "protocol": {
            "seq_len": SEQ_LEN, "feature_per_step": FEAT_PER_STEP,
            "seeds": SEEDS, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
            "bigru": "AdamW + ReduceLROnPlateau + clip 1.0 + label smoothing 0.1",
            "ftt": "AdamW, no scheduler/clip/label smoothing (same E4 protocol)",
            "note": "Each tier only changes capacity parameters; all other hyperparameters are frozen at E1.2 tuned (BiGRU) / "
                    "E2.1 tuned (FTT); no per-tier retuning (limitations see report section 5)",
            "metric_note": "head/torso/tail use the E5 primary convention; gap = train - val",
        },
        "configs": CONFIGS,
        "capacity_scan": json.loads(scan_df.to_json(orient="records")),
        "paired": paired,
        "verdicts": verdicts,
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    make_plot(scan_df, summary)
    write_report(runs, scan_df, summary)
    log(f"analysis complete | verdicts={ {k: v['verdict'] for k, v in verdicts.items()} }")
    return summary


# ==================== Plotting ====================
VERDICT_EN = {
    "Capacity sufficient, data-limited (val plateaus while the train/val gap widens) ":
        "capacity sufficient / data-limited",
    "Capacity bottleneck (Large still improves significantly; the underfitting conclusion must be corrected) ":
        "capacity bottleneck (large still improves)",
    "Suspected capacity bottleneck (improves but 95% CI spans 0; directional evidence) ":
        "possible bottleneck (directional only)",
    "Inconclusive (intermediate state; neither criterion met) ":
        "inconclusive (intermediate)",
}


def make_plot(scan_df, summary):
    if scan_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    names = {"BiGRU": "BiGRU", "FTT": "FT-Transformer"}
    for ax, arch in zip(axes, ("BiGRU", "FTT")):
        sub = scan_df[scan_df["arch"] == arch].copy()
        if sub.empty:
            ax.set_visible(False)
            continue
        sub["cap_order"] = sub["capacity"].map({c: i for i, c in enumerate(CAPACITIES)})
        sub = sub.sort_values("cap_order")
        x = sub["n_params"].values
        ax.errorbar(x, sub["val_macro_f1"] * 100,
                    yerr=sub["val_macro_f1_std"] * 100, marker="o", capsize=3,
                    color="#1f77b4", label="val Macro-F1")
        ax.errorbar(x, sub["train_macro_f1"] * 100,
                    yerr=sub["train_macro_f1_std"] * 100, marker="s", capsize=3,
                    linestyle="--", color="#d62728", label="train Macro-F1")
        for _, r in sub.iterrows():
            ax.annotate(f"{r['capacity']}", (r["n_params"], r["val_macro_f1"] * 100),
                        textcoords="offset points", xytext=(0, 11),
                        ha="center", fontsize=8)
            ax.annotate(f"gap {r['gap_macro_f1']*100:+.1f} pp",
                        (r["n_params"], r["val_macro_f1"] * 100),
                        textcoords="offset points", xytext=(0, -24),
                        ha="center", fontsize=7.5, color="#1f77b4")
        ax.set_xscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v/1e6:.2f}M" if v >= 1e6 else f"{v/1e3:.0f}K" for v in x],
                           fontsize=8)
        ax.minorticks_off()
        ax.set_xlim(x.min() / 1.7, x.max() * 1.7)
        lo = float(sub["val_macro_f1"].min()) * 100
        hi = float(sub["train_macro_f1"].max()) * 100
        ax.set_ylim(lo - 14, hi + 4)
        ax.set_xlabel("parameters (log scale)")
        ax.set_ylabel("Macro-F1 (%)")
        ax.set_title(f"{names[arch]}: capacity vs Macro-F1 (5 seeds)")
        ax.grid(alpha=0.3)
        v = summary["verdicts"].get(arch, {}).get("verdict", "—")
        ax.text(0.02, 0.98, f"verdict: {VERDICT_EN.get(v, v)}", transform=ax.transAxes,
                fontsize=8, va="top")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG, dpi=160)
    plt.close(fig)


# ==================== Report ====================
def md_table(df, cols, digits=4):
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [head, sep]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                cells.append(f"{v * 100:.2f}" if abs(v) <= 1.5 else f"{v:.2f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def fmt_d(d, col="val_macro_f1"):
    if not d or col not in d or d[col] is None:
        return "—"
    x = d[col]
    return (f"{x['mean_delta']*100:+.2f} pp "
            f"[{x['ci95_low']*100:+.2f}, {x['ci95_high']*100:+.2f}]")


def write_report(runs, scan_df, summary):
    L = []
    L.append("# E7 model capacity sweep report")
    L.append("")
    L.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')} | "
             "Companion: DeepLearning_Supplementary_Plan.md section 8 ")
    L.append("")
    L.append("## 1. Purpose and protocol")
    L.append("")
    L.append("- **Purpose**: answer report §5.2.2 - whether BiGRU / FT-Transformer's performance bottleneck is insufficient capacity,"
             "or capacity is already sufficient and limited by data scale. ")
    L.append("- **Subjects**: BiGRU (E1.2 tuned hyperparams) and FT-Transformer (E2.1 tuned hyperparams),"
             "Three capacity tiers x 5 seeds each. ")
    L.append("- **Capacity tuning**: BiGRU varies hidden_size / num_layers (sequence layout fixed at 26x4);"
             "FTT varies d_token / n_blocks. Note rtdl 0.0.13's `make_baseline` signature does not include "
             "attention_n_heads; the n_heads passed by the script (8/4/8) is filtered out by the signature, so the effective head count stays at the default, "
             "The three tiers' capacity is driven only by d_token / n_blocks (the medium tier's 489,735 parameters exactly match E2.1/E4, which proves it). ")
    L.append("- **Training**: BiGRU AdamW + ReduceLROnPlateau + clip 1.0 + label smoothing 0.1;"
             "FTT AdamW with no scheduler/clip/label smoothing (same E4 protocol, keeping the medium tier comparable); "
             "early stopping monitors val Macro-F1, 100 epochs / patience 15; seeds {42,123,777,2026,3407}. ")
    L.append("- **Metrics**: test uses Accuracy / Macro-F1 / Macro-F1(sup) / Weighted-F1 / Top-5"
             "(head/torso/tail per the E5 primary convention); capacity uses train / val Macro-F1 and "
             "gap = train − val.")
    L.append("")
    L.append("## 2. Capacity configs (measured parameter counts)")
    L.append("")
    cfg_rows = []
    for arch in ("BiGRU", "FTT"):
        for cap in CAPACITIES:
            c = dict(CONFIGS[arch][cap])
            if arch == "FTT":
                c["n_heads"] = "filtered (default 8)"
            sub = scan_df[(scan_df["arch"] == arch) & (scan_df["capacity"] == cap)]
            cfg_rows.append({
                "arch": arch, "capacity": cap,
                "config": ", ".join(f"{k}={v}" for k, v in c.items()),
                "n_params": int(sub["n_params"].iloc[0]) if not sub.empty else "—",
                "batch": int(sub["batch_used"].iloc[0]) if not sub.empty else "—",
                "mean_epochs": (f"{sub['n_epochs_mean'].iloc[0]:.0f}"
                                if not sub.empty else "—"),
                "mean_duration_min": (f"{sub['duration_s_mean'].iloc[0]/60:.1f}"
                                      if not sub.empty else "—"),
            })
    L.append(md_table(pd.DataFrame(cfg_rows),
                      ["arch", "capacity", "config", "n_params", "batch",
                       "mean_epochs", "mean_duration_min"]))
    L.append("")
    L.append("## 3. Capacity sweep results (5-seed mean ± std)")
    L.append("")
    L.append(md_table(scan_df, ["arch", "capacity", "n_params",
                                "train_macro_f1", "train_macro_f1_std",
                                "val_macro_f1", "val_macro_f1_std",
                                "gap_macro_f1", "gap_macro_f1_std",
                                "macro_f1", "macro_f1_support", "acc", "top5"]))
    L.append("")
    L.append("## 4. Paired differences and verdict")
    L.append("")
    for arch in ("BiGRU", "FTT"):
        peri = summary["paired"].get(arch)
        if not peri:
            continue
        L.append(f"### {arch}")
        L.append("")
        d_rows = []
        for cmp_name, d in peri.items():
            d_rows.append({
                "comparison": cmp_name.replace("_minus_", " − "),
                "d_val_macro_f1": fmt_d(d, "val_macro_f1"),
                "d_gap_macro_f1": fmt_d(d, "gap_macro_f1"),
                "d_test_macro_f1": fmt_d(d, "macro_f1"),
                "d_test_acc": fmt_d(d, "acc"),
            })
        L.append(md_table(pd.DataFrame(d_rows),
                          ["comparison", "d_val_macro_f1", "d_gap_macro_f1",
                           "d_test_macro_f1", "d_test_acc"]))
        L.append("")
        v = summary["verdicts"][arch]
        L.append(f"**Verdict: {v['verdict']}**")
        L.append("")
    L.append("Criterion (plan §8.2, predefined): Large - Medium Δval <= +0.5 pp and "
             "Delta gap >= +2.0 pp -> “capacity sufficient, data-limited”; delta val >= +1.0 pp and CI lower bound > 0 -> "
             "“capacity bottleneck”; delta val >= +1.0 pp but CI spans 0 -> directional evidence; otherwise inconclusive. ")
    L.append("")
    L.append(f"![capacity]({os.path.basename(FIG)})")
    L.append("")
    L.append("## 5. Evidence boundaries")
    L.append("")
    L.append("1. Only capacity parameters change per level; other hyperparameters are frozen to existing tuned values, **no per-level retuning**;"
             "Hence the Large tier may not sit at its own optimal hyperparameters, and the capacity effect could be under- or over-estimated. ")
    L.append("2. The train/val gap is measured on the training set with “best early-stopping-epoch weights”,"
             "Training-set metrics themselves do not participate in early stopping, but model selection still follows val. ")
    L.append("3. The two architectures use different protocols (FTT has no label smoothing, see E4 protocol),"
             "Hence capacity-curve heights **cannot be compared across architectures**, only tier trends within each architecture. ")
    L.append("4. This run used an 8 GB GPU with FTT batch=256 for all three capacities (no OOM downgrades),"
             "Consistent with the medium-tier convention of E2.1/E4; the archived run where the large tier was forced down to 64 on a 2 GB GPU "
             "(`archive_2GB_batch64/`) is kept as a historical record only and excluded from these statistics. ")
    L.append("5. 5 seeds (paired CI df = 4), conditioned on the single frozen test set,"
             "values may change under a different test set. ")
    L.append("")
    L.append("## 6. File list")
    L.append("")
    L.append("| File | Content |")
    L.append("|---|---|")
    L.append(f"| {os.path.basename(__file__)} | script: training (resumable) + analysis + report |")
    L.append(f"| {os.path.basename(RUN_CSV)} | full metrics per (arch, capacity, seed) |")
    L.append(f"| {os.path.basename(SCAN_CSV)} | capacity aggregation table (with train/val gap) |")
    L.append(f"| {os.path.basename(SUMMARY_JSON)} | machine-readable summary (protocol/paired differences/verdict) |")
    L.append(f"| {os.path.basename(FIG)} | capacity curve chart |")
    L.append(f"| preds/ | test predictions per (arch, capacity, seed) (int16) |")
    L.append(f"| {os.path.basename(LOG_PATH)} | run log |")
    L.append("")
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


# ==================== Main pipeline ====================
def main():
    global LOG, EPOCH_CAP, RUN_TAG
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--epoch-cap", type=int, default=None)
    ap.add_argument("--archs", type=str, default="BiGRU,FTT")
    ap.add_argument("--capacities", type=str, default="")
    ap.add_argument("--seeds", type=str, default="")
    ap.add_argument("--ftt-batch", type=int, default=None,
                    help="override the three FTT batch tiers (default 256; with 8GB VRAM can set 512 to align with E2.1/E4)")
    args = ap.parse_args()

    EPOCH_CAP = args.epoch_cap
    RUN_TAG = "smoke" if EPOCH_CAP else ""
    init_paths(RUN_TAG)
    LOG = make_logger()
    archs = [a for a in args.archs.split(",") if a]
    caps = [c for c in args.capacities.split(",") if c] or CAPACITIES
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else SEEDS

    log("=" * 72)
    log(f"E7 capacity sweep | archs={archs} | capacities={caps} | seeds={seeds} | "
        f"epoch_cap={EPOCH_CAP} | device={ec.DEVICE}")

    if not args.analyze_only:
        specs = build_specs(args.ftt_batch)
        data = load_data()
        done = load_done()
        for arch in archs:
            for cap in caps:
                if cap not in specs[arch]:
                    raise SystemExit(f"unknown capacity tier: {arch}/{cap}")
                log(f"{arch}/{cap} config: {json.dumps(specs[arch][cap]['cap_kwargs'])} | "
                    f"batch={specs[arch][cap]['params']['batch_size']}")
        for arch in archs:
            for cap in caps:
                for s in seeds:
                    run_one(arch, cap, specs[arch][cap], s, data, done)

    if os.path.exists(RUN_CSV):
        analyze()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    # On this machine under WDDM, CUDA context teardown occasionally crashes natively (0xC0000409); artifacts are already persisted,
    # Use os._exit to skip interpreter teardown, avoiding background tasks being misreported as failures
    os._exit(0)
