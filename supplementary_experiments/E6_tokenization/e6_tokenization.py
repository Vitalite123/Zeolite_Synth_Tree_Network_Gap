"""
E6 serialization and tokenization ablation
=====================================================================================
Companion documents: CBUTransfer/DeepLearning/DeepLearning_Supplementary_Plan.md §7 (E6)

E6.1 granularity ablation:
  BiGRU inputs organized as {104x1, 52x2, 26x4, 13x8}, other hyperparameters frozen to E1.2 tuned values.
  The 26x4 form is fully equivalent to E1.2/E4's BiGRU (same architecture); E4's 5-seed predictions are reused directly.

E6.2 grouping-prior ablation:
  A chemical grouping (= current 26x4, features grouped row-major in FEATURES_104 order) - reuses E6.1's 26x4;
  B random permutation: randomly permute the 104 feature columns then group 26x4 as before, 5 permutations x 5 seeds;
  C learned grouping: FT-Transformer FeatureTokenizer (independent token per feature), reuses E4's FTT 5-seed predictions.

Training protocol (consistent with E4/E5):
  AdamW + ReduceLROnPlateau + grad clip 1.0 + label smoothing 0.1;
  early stopping monitors val Macro-F1, 100 epochs / patience 15; seeds {42,123,777,2026,3407}.

Metric conventions:
  Accuracy / Macro-F1 / Macro-F1(sup) / Weighted-F1 / Top-5,
  head/torso/tail use the primary metric established in E5 (per-class F1 on the full test set -> macro average over supported classes within the group).

Usage:
  py -B e6_tokenization.py                     # train + analyze
  py -B e6_tokenization.py --analyze-only      # recompute from written predictions and generate report only
  py -B e6_tokenization.py --epoch-cap 2       # smoke test (artifacts get _smoke suffix)
  py -B e6_tokenization.py --experiments granularity --configs g13x8 --seeds 42
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
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import t as tdist
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

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

# ==================== Constants ====================
SEEDS = [42, 123, 777, 2026, 3407]
N_CLASSES = ec.N_CLASS_EXPECT
MAX_EPOCHS = 100
PATIENCE = 15
LABEL_SMOOTHING = 0.1
HEAD_MIN, TORSO_MIN = 1000, 100

GRAN = {"g104x1": (104, 1), "g52x2": (52, 2), "g26x4": (26, 4), "g13x8": (13, 8)}
REF_GRAN = "g26x4"
PERM_IDS = [101, 102, 103, 104, 105]
PERM_CSV = "E6_permutations.json"

RESULT_FIELDS = [
    "experiment", "config", "perm_id", "seed", "seq_len", "feature_per_step",
    "acc", "macro_f1", "macro_f1_support", "weighted_f1", "top5",
    "head_f1", "torso_f1", "tail_f1",
    "val_acc", "val_macro_f1",
    "n_params", "best_epoch", "duration_s", "source",
]
METRIC_COLS = ["acc", "macro_f1", "macro_f1_support", "weighted_f1", "top5",
               "head_f1", "torso_f1", "tail_f1"]

RUN_CSV = SUMMARY_GRAN = SUMMARY_GROUP = SUMMARY_JSON = REPORT_MD = None
FIG_GRAN = FIG_GROUP = PRED_DIR = LOG_PATH = None
LOG = None
EPOCH_CAP = None
RUN_TAG = ""


def init_paths(tag=""):
    global RUN_CSV, SUMMARY_GRAN, SUMMARY_GROUP, SUMMARY_JSON, REPORT_MD
    global FIG_GRAN, FIG_GROUP, PRED_DIR, LOG_PATH
    stamp = f"_{tag}" if tag else ""
    RUN_CSV = os.path.join(HERE, f"E6_runs{stamp}.csv")
    SUMMARY_GRAN = os.path.join(HERE, f"E6_granularity{stamp}.csv")
    SUMMARY_GROUP = os.path.join(HERE, f"E6_grouping_prior{stamp}.csv")
    SUMMARY_JSON = os.path.join(HERE, f"E6_summary{stamp}.json")
    REPORT_MD = os.path.join(HERE, f"E6_report{stamp}.md")
    FIG_GRAN = os.path.join(HERE, f"E6_granularity{stamp}.png")
    FIG_GROUP = os.path.join(HERE, f"E6_grouping_prior{stamp}.png")
    PRED_DIR = os.path.join(HERE, f"preds{stamp}")
    LOG_PATH = os.path.join(HERE, f"e6_run{stamp}.log")
    os.makedirs(PRED_DIR, exist_ok=True)


def log(msg):
    if LOG is not None:
        LOG.info(msg)
    else:
        print(msg, flush=True)


def make_logger():
    import logging
    lg = logging.getLogger("E6")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
        for h in (logging.FileHandler(LOG_PATH, encoding="utf-8"),
                  logging.StreamHandler()):
            h.setFormatter(fmt)
            lg.addHandler(h)
    return lg


# ==================== Model: configurable sequence granularity ====================
class BiGRUSeq(nn.Module):
    """Generic version of E1.2 BiGRU: (seq_len, feature_per_step) is determined by constructor arguments.
    (26, 4) it is layer-wise equivalent to e1_2_bigru.BiGRU_Zeolite (1,180,008 params)."""

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


def _forward(model, xb):
    return model(xb)


def load_tuned_bigru():
    with open(os.path.join(E1_DIR, "best_params_E1_2.json"), "r", encoding="utf-8") as f:
        return json.load(f)["params"]


def build_model(params, seq_len, fpp):
    return BiGRUSeq(num_classes=N_CLASSES,
                    hidden_size=int(params["hidden_size"]),
                    num_layers=int(params["num_layers"]),
                    dropout=float(params["dropout_rate"]),
                    seq_len=seq_len, feature_per_step=fpp)


# ==================== Grouping and evaluation (primary convention, same as E5) ====================
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


# ==================== Permutation definitions ====================
def load_or_make_permutations():
    """Fixes the random seed to generate and save 5 permutations for reproducibility."""
    path = os.path.join(HERE, PERM_CSV)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return {int(k): np.asarray(v, dtype=np.int64) for k, v in d.items()}
    rng = np.random.RandomState(20261024)
    perms = {pid: rng.permutation(len(ec.FEATURES_104)).astype(np.int64)
             for pid in PERM_IDS}
    with open(path, "w", encoding="utf-8") as f:
        json.dump({str(k): v.tolist() for k, v in perms.items()}, f)
    return perms


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
def train_one(params, seq_len, fpp, perm, seed, data, tag):
    ec.set_seed(seed)
    if perm is not None:
        Xtr = data["X_train"][:, perm]
        Xval = data["X_val"][:, perm]
        Xtest = data["X_test"][:, perm]
    else:
        Xtr, Xval, Xtest = data["X_train"], data["X_val"], data["X_test"]

    ds = TensorDataset(torch.as_tensor(Xtr),
                       torch.as_tensor(data["y_train"], dtype=torch.long))
    loader = DataLoader(ds, batch_size=int(params["batch_size"]), shuffle=True,
                        num_workers=0, drop_last=False)

    model = build_model(params, seq_len, fpp).to(ec.DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=float(params["learning_rate"]),
                            weight_decay=float(params["weight_decay"]))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min",
                                                       factor=0.5, patience=5)
    ls = LABEL_SMOOTHING
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
            loss = F.cross_entropy(_forward(model, xb), yb, label_smoothing=ls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total += float(loss.item())
            n_batches += 1

        val_proba = ec.predict_proba(model, Xval, _forward)
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
    test_proba = ec.predict_proba(model, Xtest, _forward)
    val_proba = ec.predict_proba(model, Xval, _forward)
    log(f"{tag} best_epoch={best_epoch} | val MacroF1={best_f1:.4f} | "
        f"test Acc={ec.metrics_from_proba(test_proba, data['y_test'], N_CLASSES)['acc']:.4f} | "
        f"params={n_params}")
    del model
    if ec.DEVICE.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
    return dict(test_proba=test_proba, val_proba=val_proba,
                n_params=int(n_params), best_epoch=int(best_epoch))


def train_with_oom_retry(params, seq_len, fpp, perm, seed, data, tag):
    bs = int(params["batch_size"])
    if ec.DEVICE.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
    for attempt in range(3):
        p = dict(params, batch_size=bs)
        try:
            return train_one(p, seq_len, fpp, perm, seed, data, tag)
        except torch.OutOfMemoryError:
            if attempt == 2 or bs // 2 < 64:
                raise
            bs = max(64, bs // 2)
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


def pred_path(config, perm_id, seed):
    return os.path.join(PRED_DIR, f"{config}_{perm_id}_seed{seed}.npy")


def load_done():
    if not os.path.exists(RUN_CSV):
        return set()
    df = pd.read_csv(RUN_CSV)
    return {(r["config"], r["perm_id"], int(r["seed"])) for _, r in df.iterrows()
            if os.path.exists(pred_path(r["config"], r["perm_id"], int(r["seed"])))}


# ==================== Reuse E4 artifacts ====================
def e4_rows(data):
    """Reuses the E4 BiGRU (=26x4) and FTT (FeatureTokenizer) five-seed predictions."""
    res_path = os.path.join(E4_DIR, "seed_repetition_results.csv")
    if not os.path.exists(res_path):
        raise RuntimeError(f"missing E4 results table: {res_path}")
    e4 = pd.read_csv(res_path)
    params = load_tuned_bigru()
    rows = []
    specs = [
        ("granularity", REF_GRAN, "chem", "bigru", 26, 4, "BiGRU"),
        ("grouping", "g26x4", "chem", "bigru", 26, 4, "BiGRU"),
        ("grouping", "ftt_learned", "learned", "ftt", "", "", "FT-Transformer"),
    ]
    seen = set()
    for exp, cfg, pid, pslug, seq_len, fpp, model_name in specs:
        src = e4[e4["model"] == model_name].set_index("seed")
        for seed in SEEDS:
            p = os.path.join(E4_DIR, "preds", f"{pslug}_seed{seed}.npy")
            if seed not in src.index or not os.path.exists(p):
                raise RuntimeError(f"missing E4 artifacts: {model_name} seed={seed}")
            if (cfg, pid, seed) in seen:
                continue
            seen.add((cfg, pid, seed))
            pred = np.load(p).astype(np.int64)
            m = evaluate_pred(pred, data["y_test"], data)
            r = src.loc[seed]
            row = {
                "experiment": exp, "config": cfg, "perm_id": pid, "seed": int(seed),
                "seq_len": seq_len, "feature_per_step": fpp,
                "top5": round(float(r["top5"]), 6),
                "val_acc": round(float(r["val_acc"]), 6),
                "val_macro_f1": round(float(r["val_macro_f1"]), 6),
                "n_params": int(r["n_params"]),
                "best_epoch": int(r["best_epoch"]),
                "duration_s": float(r["duration_s"]),
                "source": f"E4_seeds/preds/{pslug}_seed{seed}.npy",
            }
            for k in METRIC_COLS:
                if k not in row:
                    row[k] = round(float(m[k]), 6)
            rows.append(row)
            # Persist E4 predictions for 26x4 for recomputation/reuse
            if pslug == "bigru":
                np.save(pred_path(REF_GRAN, "chem", seed), pred.astype(np.int16))
    return rows


# ==================== Single run ====================
def run_one(exp, config, perm_id, seq_len, fpp, perm, seed, params, data, done):
    if (config, perm_id, seed) in done:
        log(f"[skip] {config}/{perm_id} seed={seed} already done")
        return
    tag = f"[{config}/{perm_id} seed={seed}]"
    log(f"=== {tag} start ===")
    t0 = time.time()
    out = train_with_oom_retry(params, seq_len, fpp, perm, seed, data, tag)
    pred = out["test_proba"].argmax(axis=1)
    m = evaluate_pred(pred, data["y_test"], data)
    top5 = ec.metrics_from_proba(out["test_proba"], data["y_test"], N_CLASSES)["top5"]
    val_m = ec.metrics_from_proba(out["val_proba"], data["y_val"], N_CLASSES)
    np.save(pred_path(config, perm_id, seed), pred.astype(np.int16))
    row = {
        "experiment": exp, "config": config, "perm_id": perm_id, "seed": int(seed),
        "seq_len": seq_len, "feature_per_step": fpp,
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
    append_row(RUN_CSV, row, RESULT_FIELDS)
    log(f"=== {tag} done in {row['duration_s']}s | Acc={row['acc']:.4f} "
        f"MacroF1={row['macro_f1']:.4f} Top5={row['top5']:.4f} ===")


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


def series(df, config, perm_id, col):
    sub = df[(df["config"] == config) & (df["perm_id"] == perm_id)]
    return {int(r["seed"]): float(r[col]) for _, r in sub.iterrows()}


# ==================== Analysis ====================
def analyze():
    log("=== Analysis phase ===")
    if not os.path.exists(RUN_CSV):
        raise RuntimeError(f"missing run log: {RUN_CSV}")
    runs = pd.read_csv(RUN_CSV)
    runs = (runs.drop_duplicates(subset=["config", "perm_id", "seed"], keep="last")
                .sort_values(["experiment", "config", "perm_id", "seed"])
                .reset_index(drop=True))
    runs.to_csv(RUN_CSV, index=False, encoding="utf-8-sig")

    # ---------- E6.1 granularity ----------
    ref_series = {col: series(runs, REF_GRAN, "chem", col) for col in METRIC_COLS}
    gran_rows = []
    for cfg, (seq_len, fpp) in GRAN.items():
        sub = runs[(runs["config"] == cfg) & (runs["perm_id"] == "chem")]
        if sub.empty:
            log(f"[warn] missing granularity config {cfg}, skipping")
            continue
        row = {"config": cfg, "seq_len": seq_len, "feature_per_step": fpp,
               "n_seeds": int(sub["seed"].nunique())}
        for col in METRIC_COLS:
            mu, sd = mean_std(sub[col].values)
            row[col] = round(mu, 6)
            row[f"{col}_std"] = round(sd, 6)
        row["n_params"] = int(sub["n_params"].iloc[0])
        if cfg != REF_GRAN:
            for col in ("acc", "macro_f1"):
                d = paired_delta(series(runs, cfg, "chem", col), ref_series[col])
                row[f"d_{col}_vs_{REF_GRAN}"] = round(d["mean_delta"], 6)
                row[f"d_{col}_ci_low"] = round(d["ci95_low"], 6)
                row[f"d_{col}_ci_high"] = round(d["ci95_high"], 6)
        gran_rows.append(row)
    gran_df = pd.DataFrame(gran_rows)
    # Column order: config, seq_len, fpp, n_seeds, acc.., std.., n_params, deltas..
    gran_df.to_csv(SUMMARY_GRAN, index=False, encoding="utf-8-sig")

    # ---------- E6.2 grouping prior ----------
    perm_ids = sorted([p for p in runs["perm_id"].dropna().unique()
                       if str(p).startswith("perm")])
    group_rows = []

    def _add_group_row(scheme, detail, sub, ref_series_pair):
        row = {"scheme": scheme, "detail": detail,
               "n_runs": int(len(sub)), "n_seeds": int(sub["seed"].nunique())}
        for col in ("acc", "macro_f1", "macro_f1_support", "weighted_f1", "top5"):
            mu, sd = mean_std(sub[col].values)
            row[col] = round(mu, 6)
            row[f"{col}_std"] = round(sd, 6)
        for col in ("acc", "macro_f1"):
            d = ref_series_pair.get(col)
            if d:
                row[f"d_{col}_vs_chem"] = round(d["mean_delta"], 6)
                row[f"d_{col}_ci_low"] = round(d["ci95_low"], 6)
                row[f"d_{col}_ci_high"] = round(d["ci95_high"], 6)
        group_rows.append(row)

    chem_sub = runs[(runs["config"] == REF_GRAN) & (runs["perm_id"] == "chem")]
    _add_group_row("A_chem", "FEATURES_104 order, 26x4", chem_sub, {})

    perm_vals = {}
    for pid in perm_ids:
        sub = runs[(runs["config"] == REF_GRAN) & (runs["perm_id"] == pid)]
        d = {col: paired_delta(series(runs, REF_GRAN, pid, col), ref_series[col])
             for col in ("acc", "macro_f1")}
        _add_group_row(f"B_{pid}", f"random permutation {pid}", sub, d)
        for col in ("acc", "macro_f1"):
            for s, v in series(runs, REF_GRAN, pid, col).items():
                perm_vals.setdefault((col, s), []).append(v)
    # Random-permutation merge: per seed, average over the 5 permutations first, then pair with chem
    perm_pooled = {}
    for col in ("acc", "macro_f1"):
        a = {s: float(np.mean(v)) for (c, s), v in perm_vals.items() if c == col}
        perm_pooled[col] = paired_delta(a, ref_series[col])
    perm_sub = runs[(runs["config"] == REF_GRAN)
                    & (runs["perm_id"].astype(str).str.startswith("perm"))]
    _add_group_row("B_random_pooled", f"{len(perm_ids)} permutations pooled",
                   perm_sub, perm_pooled)

    learned_sub = runs[runs["perm_id"] == "learned"]
    if not learned_sub.empty:
        d = {col: paired_delta(series(runs, "ftt_learned", "learned", col),
                               ref_series[col]) for col in ("acc", "macro_f1")}
        _add_group_row("C_learned_ftt", "FT-Transformer FeatureTokenizer", learned_sub, d)
    group_df = pd.DataFrame(group_rows)
    group_df.to_csv(SUMMARY_GROUP, index=False, encoding="utf-8-sig")

    # ---------- Verdict ----------
    # 26x4 vs 13x8 paired differences computed directly from runs (the table records each config relative to 26x4)
    d_g26_g13 = {col: paired_delta(series(runs, REF_GRAN, "chem", col),
                                   series(runs, "g13x8", "chem", col))
                 for col in ("acc", "macro_f1")}
    d_perm = perm_pooled
    # Permutation worse than chemical ordering => the order prior is effective (the difference is defined as perm - chem, so the upper bound must be < 0)
    prior_effect = [k for k in ("acc", "macro_f1")
                    if d_perm.get(k) and d_perm[k]["mean_delta"] < 0
                    and d_perm[k]["ci95_high"] < 0]
    gran_effect = [k for k in ("acc", "macro_f1")
                   if d_g26_g13[k] and d_g26_g13[k]["mean_delta"] >= 0.02]
    if len(gran_effect) == 2 and len(prior_effect) >= 1:
        verdict = "serialization/grouping prior dominates"
    elif not gran_effect and not prior_effect:
        verdict = "circular inductive bias dominates (specific granularity/order yields no gain)"
    else:
        verdict = "mixed: partly from granularity priors, partly from circular inductive bias"

    summary = {
        "experiment": "E6_tokenization",
        "date": "2026-10-24",
        "protocol": {
            "model": "BiGRU (E1.2 tuned) / FT-Transformer (E2.1 tuned, E4 reused)",
            "seeds": SEEDS, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
            "label_smoothing": LABEL_SMOOTHING,
            "granularities": {k: list(v) for k, v in GRAN.items()},
            "permutation_ids": PERM_IDS,
            "metric_note": "head/torso/tail use the E5 primary convention (per-class F1 over the full test set -> macro average over supported classes within the group)",
        },
        "granularity": json.loads(gran_df.to_json(orient="records")),
        "grouping": json.loads(group_df.to_json(orient="records")),
        "paired": {
            "d_26x4_minus_13x8": d_g26_g13,
            "d_chem_minus_random_perms": d_perm,
            "d_learned_ftt_minus_chem": {
                col: paired_delta(series(runs, "ftt_learned", "learned", col),
                                  ref_series[col])
                for col in ("acc", "macro_f1")} if not learned_sub.empty else {},
        },
        "verdict": verdict,
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    make_gran_plot(gran_df)
    make_group_plot(group_df)
    write_report(runs, gran_df, group_df, summary)
    log(f"analysis complete | verdict={verdict}")
    return summary


def make_gran_plot(gran_df):
    if gran_df.empty:
        return
    df = gran_df.sort_values("feature_per_step")
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.errorbar(x, df["acc"] * 100, yerr=df["acc_std"] * 100, marker="o",
                capsize=3, label="Accuracy")
    ax.errorbar(x, df["macro_f1_support"] * 100,
                yerr=df["macro_f1_support_std"] * 100, marker="s",
                capsize=3, label="Macro-F1 (support)")
    ax.errorbar(x, df["top5"] * 100, yerr=df["top5_std"] * 100, marker="^",
                capsize=3, label="Top-5")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['seq_len']}x{r['feature_per_step']}" for _, r in df.iterrows()])
    ax.set_xlabel("serialization granularity (seq_len x features_per_step)")
    ax.set_ylabel("test metric (%)")
    ax.set_title("E6.1 BiGRU serialization granularity (5 seeds)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_GRAN, dpi=160)
    plt.close(fig)


def make_group_plot(group_df):
    if group_df.empty:
        return
    labels = list(group_df["scheme"])
    acc = group_df["acc"].values * 100
    acc_sd = group_df["acc_std"].values * 100
    mac = group_df["macro_f1_support"].values * 100
    mac_sd = group_df["macro_f1_support_std"].values * 100
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.bar(x - w / 2, acc, w, yerr=acc_sd, capsize=3, label="Accuracy")
    ax.bar(x + w / 2, mac, w, yerr=mac_sd, capsize=3, label="Macro-F1 (support)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("test metric (%)")
    ax.set_title("E6.2 grouping prior: chemical order vs random permutation vs learned")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_GROUP, dpi=160)
    plt.close(fig)


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


def write_report(runs, gran_df, group_df, summary):
    p = summary["paired"]
    d_g = p["d_26x4_minus_13x8"]
    d_p = p["d_chem_minus_random_perms"]
    d_c = p.get("d_learned_ftt_minus_chem", {})

    def fmt_d(d, col="acc"):
        if not d or col not in d or d[col] is None:
            return "—"
        x = d[col]
        return (f"{x['mean_delta']*100:+.2f} pp "
                f"[{x['ci95_low']*100:+.2f}, {x['ci95_high']*100:+.2f}]")

    L = []
    L.append("# E6 serialization and tokenization ablation report")
    L.append("")
    L.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')} | "
             "Companion: DeepLearning_Supplementary_Plan.md section 7 ")
    L.append("")
    L.append("## 1. Purpose and protocol")
    L.append("")
    L.append("- **Purpose**: answer report §5.6.3 - how much of BiGRU's advantage comes from the 26x4 serialization/grouping prior,"
             "How much comes from recurrence inductive bias. ")
    L.append("- **E6.1 granularity**: BiGRU inputs organized as {104x1, 52x2, 26x4, 13x8},"
             "Hyperparameters are frozen at E1.2 tuned values; 26x4 is layer-wise equivalent to the E1.2/E4 architecture, directly reusing E4's five-seed predictions. ")
    L.append("- **E6.2 grouping prior**: A chemical order (FEATURES_104 row-major grouping, i.e., the status quo);"
             "B Random permutation: permute feature columns randomly but still group 26x4 (5 fixed permutations); "
             "C learned grouping -- FT-Transformer FeatureTokenizer (reuses the E4 FTT five-seed predictions).")
    L.append("- **Training**: AdamW + ReduceLROnPlateau + grad clip 1.0 + label smoothing 0.1;"
             "early stopping monitors val Macro-F1, 100 epochs / patience 15; seeds {42,123,777,2026,3407}. ")
    L.append("- **Metrics**: Accuracy / Macro-F1 / Macro-F1(sup) / Weighted-F1 / Top-5;"
             "head/torso/tail use the primary convention established by E5. ")
    L.append("")
    L.append("## 2. E6.1 granularity ablation (5-seed mean ± std)")
    L.append("")
    L.append(md_table(gran_df, ["config", "seq_len", "feature_per_step", "n_params",
                               "acc", "acc_std", "macro_f1", "macro_f1_std",
                               "macro_f1_support", "weighted_f1", "top5"]))
    L.append("")
    L.append("Differences (vs 26x4, paired seeds, 95% CI; d_macro is the default Macro-F1 convention):")
    L.append("")
    acc_key = f"d_acc_vs_{REF_GRAN}"
    if acc_key in gran_df.columns:
        rows = []
        for _, r in gran_df.iterrows():
            if r["config"] == REF_GRAN:
                continue
            rows.append({
                "config": r["config"],
                "d_acc": (f"{r[acc_key]*100:+.2f} pp "
                          f"[{r['d_acc_ci_low']*100:+.2f}, {r['d_acc_ci_high']*100:+.2f}]"),
                "d_macro": (f"{r[f'd_macro_f1_vs_{REF_GRAN}']*100:+.2f} pp "
                            f"[{r['d_macro_f1_ci_low']*100:+.2f}, "
                            f"{r['d_macro_f1_ci_high']*100:+.2f}]"),
            })
        L.append(md_table(pd.DataFrame(rows), ["config", "d_acc", "d_macro"]))
    L.append("")
    L.append(f"- 26×4 vs 13×8:ΔAcc {fmt_d(d_g)};ΔMacro-F1 {fmt_d(d_g, 'macro_f1')}.")
    L.append("")
    L.append("## 3. E6.2 grouping-prior ablation")
    L.append("")
    L.append(md_table(group_df, ["scheme", "detail", "n_runs", "n_seeds", "acc", "acc_std",
                                 "macro_f1", "macro_f1_support", "weighted_f1", "top5"]))
    L.append("")
    L.append(f"- random permutation vs chemical order (per seed, average the 5 permutations first, then pair; negative = permutation worse):"
             f"ΔAcc {fmt_d(d_p)};ΔMacro-F1 {fmt_d(d_p, 'macro_f1')}.")
    L.append(f"- learned FeatureTokenizer vs chemical order (negative = FTT lower): ΔAcc {fmt_d(d_c)};"
             f"ΔMacro-F1 {fmt_d(d_c, 'macro_f1')}.")
    L.append("")
    L.append(f"![granularity]({os.path.basename(FIG_GRAN)})")
    L.append("")
    L.append(f"![grouping]({os.path.basename(FIG_GROUP)})")
    L.append("")
    L.append("## 4. Conclusions and evidence boundaries")
    L.append("")
    L.append(f"**Verdict: {summary['verdict']}**")
    L.append("")
    L.append("1. Verdict rule (plan §7.2): if the Acc gain of 26x4 over 13x8 is >=2 pp and random permutation is significantly worse than chemical order,"
             "attribute to “serialization/grouping-prior dominance”; if the granularity difference is <2 pp and permutation vs chemical ordering is not significant, "
             "attribute to “recurrence inductive bias dominance”. ")
    if not gran_df.empty:
        b = gran_df.loc[gran_df["acc"].idxmax()]
        w = gran_df.loc[gran_df["acc"].idxmin()]
        L.append(f"2. Granularity effect magnitude: the four-capacity Acc range is only {(b['acc']-w['acc'])*100:.2f} pp"
                 f"({b['config']} highest {b['acc']*100:.2f}%, {w['config']} lowest {w['acc']*100:.2f}%);"
                 "26x4 is not the optimal config, and the 26x4 vs 13x8 delta Acc is only "
                 f"{(d_g.get('acc', {}) or {}).get('mean_delta', float('nan'))*100:+.2f} pp."
                 "The hypothesis that “a finer 26x4 partition brings gains” does not hold. ")
    L.append("3. Evidence boundary: 5 permutations only coarsely sample the “order space” and cannot represent the full permutation distribution;"
             "Random permutation destroys both chemical adjacency and feature order, so their individual contributions cannot be separated; "
             "Option C (FTT) differs from BiGRU in parameter count and training protocol, serving only as a “sequence-free” reference; "
             "Cannot be interpreted as a controlled comparison at equal capacity. ")
    L.append("4. This experiment changes only the input organization, not the data, splits, label space or loss;"
             "The granularity effect is measured within a single architecture (BiGRU tuned); extrapolation to LeNet/Transformer requires caution. ")
    L.append("5. Only 5 seeds (paired CI df = 4), and per-seed paired differences are conditioned on the test set,"
             "may change under a different test set. ")
    L.append("")
    L.append("## 5. File list")
    L.append("")
    L.append("| File | Content |")
    L.append("|---|---|")
    L.append(f"| {os.path.basename(__file__)} | script: training (resumable) + analysis + report |")
    L.append(f"| {os.path.basename(RUN_CSV)} | full metrics per (config, perm_id, seed) |")
    L.append(f"| {os.path.basename(SUMMARY_GRAN)} | E6.1 granularity aggregation (with paired differences) |")
    L.append(f"| {os.path.basename(SUMMARY_GROUP)} | E6.2 grouping-prior aggregation (with paired differences) |")
    L.append(f"| {os.path.basename(SUMMARY_JSON)} | machine-readable summary (protocol/paired differences/verdict) |")
    L.append(f"| {os.path.basename(FIG_GRAN)} | granularity curve chart |")
    L.append(f"| {os.path.basename(FIG_GROUP)} | grouping-prior comparison chart |")
    L.append(f"| {os.path.basename(PERM_CSV)} | column indices of the 5 random permutations (reproducible) |")
    L.append(f"| preds/ | test predictions per (config, perm_id, seed) (int16) |")
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
    ap.add_argument("--experiments", type=str, default="granularity,grouping")
    ap.add_argument("--configs", type=str, default="")
    ap.add_argument("--perms", type=str, default="")
    ap.add_argument("--seeds", type=str, default="")
    args = ap.parse_args()

    EPOCH_CAP = args.epoch_cap
    RUN_TAG = "smoke" if EPOCH_CAP else ""
    init_paths(RUN_TAG)
    LOG = make_logger()
    experiments = [e for e in args.experiments.split(",") if e]
    configs = [c for c in args.configs.split(",") if c] or list(GRAN)
    perms = [int(p) for p in args.perms.split(",") if p] or PERM_IDS
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else SEEDS

    log("=" * 72)
    log(f"E6 serialization/tokenization ablation | experiments={experiments} | configs={configs} | "
        f"perms={perms} | seeds={seeds} | epoch_cap={EPOCH_CAP} | device={ec.DEVICE}")

    if not args.analyze_only:
        params = load_tuned_bigru()
        log(f"BiGRU tuned parameters: {json.dumps(params)}")
        data = load_data()
        permutations = load_or_make_permutations()

        existing = set()
        if os.path.exists(RUN_CSV):
            prev = pd.read_csv(RUN_CSV)
            existing = {(r["config"], r["perm_id"], int(r["seed"]))
                        for _, r in prev.iterrows()}
        for r in e4_rows(data):
            if (r["config"], r["perm_id"], r["seed"]) not in existing:
                append_row(RUN_CSV, r, RESULT_FIELDS)
        log("E4 reuse complete: 26x4 chem(5 seeds) + FTT learned(5 seeds)")

        done = load_done()
        if "granularity" in experiments:
            for cfg in configs:
                if cfg == REF_GRAN:
                    continue
                seq_len, fpp = GRAN[cfg]
                for s in seeds:
                    run_one("granularity", cfg, "chem", seq_len, fpp, None,
                            s, params, data, done)
        if "grouping" in experiments:
            for pid in perms:
                perm = permutations[int(pid)]
                seq_len, fpp = GRAN[REF_GRAN]
                for s in seeds:
                    run_one("grouping", REF_GRAN, f"perm{pid}", seq_len, fpp,
                            perm, s, params, data, done)

    if os.path.exists(RUN_CSV):
        analyze()


if __name__ == "__main__":
    main()
