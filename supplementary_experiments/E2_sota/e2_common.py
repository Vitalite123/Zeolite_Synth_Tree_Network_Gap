"""
E2 common module: data loading / evaluation / rtdl training loop / artifact export
================================================================
Companion documents: CBUTransfer/DeepLearning/DeepLearning_Supplementary_Plan.md §2/§3 (E2.1-E2.3)
Protocol alignment: fully consistent with E1_4 (data/split/label encoding/fillna(0)/seed/metrics/schema)

Key conventions:
  - Features: FEATURES_104, fillna(0); deep models additionally use StandardScaler (fitted on train only, protocol §1.2)
  - Labels: Code1_index -> LabelEncoder (231 classes); top-k labels must use integer indices np.arange(n)
  - Seeds: 42; early stopping: monitor val Macro-F1, patience 15 (deep models)
  - Objective: val Macro-F1; TPE(seed=42) + MedianPruner
  - Device: prefer cuda (MX450 2GB); clear GPU memory after each trial
"""

import copy
import json
import logging
import os
import time
import warnings

# MX450 has only 2GB VRAM: reduce fragmentation (must be set before importing torch)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import optuna
import pandas as pd
import gc
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, top_k_accuracy_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# pandas 3.0 defaults to the pyarrow string backend (string_storage='auto'); on this machine with Python 3.14 +
# under the pyarrow combination, `pd.DataFrame(rows)` triggers a native access violation (0xC0000005,
# top of stack: pandas/core/arrays/string_arrow.py:_from_sequence). Force fallback to the python storage.
try:
    pd.set_option("mode.string_storage", "python")
except Exception:
    pass

# ==================== Paths and constants (same as E1_4) ====================
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))  # Git/ (repo root holding data_processed/)
DATA_DIR = os.path.join(ROOT, "data_processed")
TRAIN_PATH = os.path.join(DATA_DIR, "ZEOSYN_104_struct_train.xlsx")
VAL_PATH = os.path.join(DATA_DIR, "ZEOSYN_104_struct_val.xlsx")
TEST_PATH = os.path.join(DATA_DIR, "ZEOSYN_104_struct_test.xlsx")

SEED = 42
N_CLASS_EXPECT = 231
MAX_EPOCHS = 100          # deep-model cap, protocol §1.4
PATIENCE = 15             # early-stopping patience, protocol §1.4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 104-feature definitions (consistent with E1_4/Baseline104/bigru_104, copied column by column)
FEATURES_104 = [
    "Si", "Al", "P", "Na", "K", "Li", "Sr", "Rb", "Cs", "Ba", "Ca", "F", "Ge",
    "Ti", "In", "B", "Mg", "Ga", "Ni", "Mn", "Fe", "Co", "Cr", "Zn", "Nb", "Be",
    "W", "Ce", "Cu", "Sn", "Gd", "La", "Y", "Dy", "Sm", "Ag", "Cd", "Zr", "V",
    "Ta", "Ru", "Hf", "Yb", "Tl", "As",
    "osda1_index", "osda2_index", "osda3_index",
    "cryst_temp", "cryst_time", "seed", "rotation",
    "aging_temp", "aging_time",
    "acid", "OH",
    "H2O_T", "OH_T", "Gel_Si_Al", "Gel_P_Al", "Gel_P_Si",
    "osda1_bertz_ct_mean_0", "osda1_free_sasa_mean_0", "osda1_asphericity_mean_0",
    "osda1_eccentricity_mean_0", "osda1_axes_mean_0", "osda1_axes_mean_1",
    "osda1_box_mean_0", "osda1_box_mean_1", "osda1_box_mean_2",
    "osda1_getaway_mean_0", "osda1_getaway_mean_1",
    "osda2_bertz_ct_mean_0", "osda2_free_sasa_mean_0", "osda2_asphericity_mean_0",
    "osda2_eccentricity_mean_0", "osda2_axes_mean_0", "osda2_axes_mean_1",
    "osda2_box_mean_0", "osda2_box_mean_1", "osda2_box_mean_2",
    "osda2_getaway_mean_0", "osda2_getaway_mean_1",
    "osda3_bertz_ct_mean_0", "osda3_free_sasa_mean_0", "osda3_asphericity_mean_0",
    "osda3_eccentricity_mean_0", "osda3_axes_mean_0", "osda3_axes_mean_1",
    "osda3_box_mean_0", "osda3_box_mean_1", "osda3_box_mean_2",
    "osda3_getaway_mean_0", "osda3_getaway_mean_1",
    "osda_avg_asphericity", "osda_max_asphericity", "osda_min_asphericity",
    "osda_avg_sasa", "osda_max_sasa", "osda_min_sasa",
    "osda_avg_bertz", "osda_max_bertz", "osda_min_bertz", "osda_total_volume",
]


# ==================== Logging (file + console dual write, same as E1_4) ====================
def make_logger(name, filename):
    path = os.path.join(HERE, filename)
    lg = logging.getLogger(name)
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        _fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
        for h in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()):
            h.setFormatter(_fmt)
            lg.addHandler(h)
    return lg


# ==================== Random seeds ====================
def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==================== Long-tail stratification (report section 4.5 / Fig. S9) ====================
def make_strata(y_train, n_classes, n_groups=3):
    """Three equal parts by descending training-set class frequency (split by class count, not sample count).
    returns {"head"/"torso"/"tail": arrays of original label indices for that class group}."""
    counts = np.bincount(y_train, minlength=n_classes)
    order = np.argsort(-counts, kind="stable")
    groups = np.array_split(order, n_groups)
    return dict(zip(("head", "torso", "tail"), groups))


# ==================== Data loading (protocol sections 1.1/1.2) ====================
CACHE_PATH = os.path.join(DATA_DIR, "ZEOSYN_104_struct_cache.npz")


def load_data(logger):
    logger.info("=" * 64)
    logger.info(f"loading data (unified protocol §1.1) | device={DEVICE}")
    # npz cache preferred: content matches the xlsx cell by cell (feature selection + fillna(0)); format conversion only;
    # Avoid sporadic native crashes of pandas/openpyxl in long-lived processes (0xC0000005, reproduced twice on 2026-10-21)
    if os.path.exists(CACHE_PATH):
        z = np.load(CACHE_PATH)
        X_train, y_train = z["X_train"], z["y_train"]
        X_val, y_val = z["X_val"], z["y_val"]
        X_test, y_test = z["X_test"], z["y_test"]
        logger.info(f"loading from npz cache | train {X_train.shape} | val {X_val.shape} "
                    f"| test {X_test.shape}")
    else:
        train_df = pd.read_excel(TRAIN_PATH)
        val_df = pd.read_excel(VAL_PATH)
        test_df = pd.read_excel(TEST_PATH)
        logger.info(f"train {train_df.shape} | val {val_df.shape} | test {test_df.shape}")
        X_train = train_df[FEATURES_104].fillna(0).to_numpy(dtype=np.float32)
        y_train = train_df["Code1_index"].to_numpy()
        X_val = val_df[FEATURES_104].fillna(0).to_numpy(dtype=np.float32)
        y_val = val_df["Code1_index"].to_numpy()
        X_test = test_df[FEATURES_104].fillna(0).to_numpy(dtype=np.float32)
        y_test = test_df["Code1_index"].to_numpy()
        np.savez_compressed(CACHE_PATH, X_train=X_train, y_train=y_train,
                            X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)
        logger.info(f"xlsx read successfully, npz cache written: {CACHE_PATH}")

    feats_ok = X_train.shape[1] == len(FEATURES_104)
    if not feats_ok:
        raise RuntimeError(f"feature count {X_train.shape[1]} != {len(FEATURES_104)}")

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_val = le.transform(y_val)
    y_test = le.transform(y_test)
    n_classes = len(le.classes_)
    if n_classes != N_CLASS_EXPECT:
        raise RuntimeError(f"class count {n_classes} != expected {N_CLASS_EXPECT}")
    logger.info(f"classes: {n_classes} | features: {X_train.shape[1]} (all 104 available)")

    # Deep-model unified requirement: StandardScaler fit on train only (protocol section 1.2)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)
    logger.info("StandardScaler fitted on train and applied to val/test")

    # Long-tail stratification: three equal parts by training-set class frequency (1/3 of classes each), for head/torso/tail F1
    strata = make_strata(y_train, n_classes)
    logger.info(f"long-tail strata (train class frequency terciles): head {len(strata['head'])} classes "
                f"torso {len(strata['torso'])} classes tail {len(strata['tail'])} classes")
    return dict(
        X_train=X_train_s, y_train=y_train,
        X_val=X_val_s, y_val=y_val,
        X_test=X_test_s, y_test=y_test,
        n_classes=n_classes, strata=strata,
    )


# ==================== Evaluation ====================
def predict_proba(model, X, forward, batch=256):
    """eval batch defaults to 256: on MX450 2GB, the FTT large config (d128x5L) at batch 2048
    attention allocation (~690MB) would OOM; internally also has OOM halving protection (floor 64)."""
    model.eval()
    while True:
        try:
            outs = []
            with torch.no_grad():
                for i in range(0, len(X), batch):
                    xb = torch.as_tensor(X[i:i + batch], device=DEVICE)
                    outs.append(F.softmax(forward(model, xb), dim=1).cpu().numpy())
            return np.concatenate(outs)
        except torch.OutOfMemoryError:
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            if batch // 2 < 64:
                raise
            batch //= 2


def metrics_from_proba(proba, y, n_classes, strata=None):
    """Full metric set: Acc/Precision-Macro/Recall-Macro/F1-Macro/
    F1-Weighted, the Top-3/5/10 needed by §4.2, and the head/torso/tail stratified Macro-F1 needed by §4.5.
    strata is the return of make_strata(); stratified F1 is computed only on the sample subset corresponding to that group's classes,
    labels restricted to that group's classes, avoiding other groups' classes lowering/raising the mean."""
    pred = proba.argmax(axis=1)
    labels = np.arange(n_classes)
    out = dict(
        acc=float(accuracy_score(y, pred)),
        precision_macro=float(precision_score(y, pred, average="macro",
                                              zero_division=0)),
        recall_macro=float(recall_score(y, pred, average="macro",
                                        zero_division=0)),
        macro_f1=float(f1_score(y, pred, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y, pred, average="weighted", zero_division=0)),
        top3=float(top_k_accuracy_score(y, proba, k=3, labels=labels)),
        top5=float(top_k_accuracy_score(y, proba, k=5, labels=labels)),
        top10=float(top_k_accuracy_score(y, proba, k=10, labels=labels)),
    )
    if strata:
        for name in ("head", "torso", "tail"):
            idx = strata[name]
            mask = np.isin(y, idx)
            out[f"{name}_f1"] = (
                float("nan") if mask.sum() == 0 else
                float(f1_score(y[mask], pred[mask], labels=idx,
                               average="macro", zero_division=0)))
    return out


# ==================== rtdl training loop (with per-epoch Optuna pruning) ====================
def train_rtdl(build_fn, forward, params, data, logger, trial=None,
               max_epochs=MAX_EPOCHS, patience=PATIENCE, tag=""):
    """build_fn(params)->nn.Module;  forward(model, xb)->logits
    MX450 2GB VRAM protection: on CUDA OOM automatically halve the batch and retry (up to 2 times, floor 64).
    returns (model_best_state, metrics, curve, best_epoch)"""
    set_seed(SEED)
    n_classes = data["n_classes"]
    X_val, y_val = data["X_val"], data["y_val"]
    bs = params["batch_size"]
    # Clear the VRAM pool before each training run: avoid fragmentation from the previous trial's cached blocks causing OOM -> driver-level crash
    if DEVICE.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
    for attempt in range(3):  # OOM retry with halved batch
        try:
            return _train_rtdl_once(build_fn, forward, params, bs, data, logger,
                                    trial, max_epochs, patience, tag, attempt)
        except torch.OutOfMemoryError:
            if attempt == 2 or bs // 2 < 64:
                raise
            bs //= 2
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            logger.info(f"{tag} CUDA OOM, halving batch to {bs} and retrying")


def _train_rtdl_once(build_fn, forward, params, bs, data, logger, trial,
                     max_epochs, patience, tag, attempt):
    n_classes = data["n_classes"]
    train_ds = TensorDataset(torch.as_tensor(data["X_train"]),
                             torch.as_tensor(data["y_train"], dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=0, pin_memory=(DEVICE.type == "cuda"))
    X_val, y_val = data["X_val"], data["y_val"]

    model = build_fn(params).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=params["lr"],
                            weight_decay=params["weight_decay"])

    best_f1, best_state, best_epoch, bad = -1.0, None, 0, 0
    curve = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        total, n_batches = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            loss = F.cross_entropy(forward(model, xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.item())
            n_batches += 1
        val_f1 = metrics_from_proba(predict_proba(model, X_val, forward),
                                    y_val, n_classes)["macro_f1"]
        curve.append((epoch, total / max(n_batches, 1), val_f1))
        if trial is not None:
            trial.report(val_f1, epoch)
            if trial.should_prune():
                del model
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()
                raise optuna.TrialPruned(f"epoch {epoch}: val MacroF1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1, best_epoch, bad = val_f1, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    metrics = {
        "val": metrics_from_proba(predict_proba(model, X_val, forward), y_val,
                                  n_classes, data.get("strata")),
        "test": metrics_from_proba(predict_proba(model, data["X_test"], forward),
                                   data["y_test"], n_classes, data.get("strata")),
    }
    metrics["n_params"] = n_params
    metrics["best_epoch"] = best_epoch
    metrics["train_val_gap"] = float(metrics["val"]["acc"] - metrics["test"]["acc"])
    logger.info(f"{tag} best_epoch={best_epoch}/{len(curve)} | "
                f"val MacroF1={best_f1:.4f} | test Acc={metrics['test']['acc']:.4f} "
                f"MacroF1={metrics['test']['macro_f1']:.4f} | params={n_params}")
    if DEVICE.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
    return model, metrics, curve, best_epoch


# ==================== Artifact export (schema matches E1_4; extended with columns required by report sections 4.1/4.2/4.5) ====================
SUMMARY_COLUMNS = ["exp_id", "model", "config_id", "seed", "fraction", "n_params",
                   "train_val_gap",
                   "test_acc", "test_precision_macro", "test_recall_macro",
                   "test_macro_f1", "test_weighted_f1",
                   "test_top3", "test_top5", "test_top10",
                   "head_f1", "torso_f1", "tail_f1", "notes"]


def _r(metrics, key):
    """Extracts metrics and keeps 4 decimals; missing/NaN returns an empty string so CSV can directly backfill report tables."""
    v = metrics.get(key, float("nan"))
    if isinstance(v, str):
        return v
    return "" if (v is None or (isinstance(v, float) and np.isnan(v))) else round(float(v), 4)


def summary_row(exp_id, model, config_id, metrics, notes="", fraction=1.0):
    test = metrics.get("test", {})
    row = {
        "exp_id": exp_id, "model": model, "config_id": config_id,
        "seed": SEED, "fraction": fraction,
        "n_params": metrics.get("n_params", ""),
        "train_val_gap": _r(metrics, "train_val_gap"),
        "test_acc": _r(test, "acc"),
        "test_precision_macro": _r(test, "precision_macro"),
        "test_recall_macro": _r(test, "recall_macro"),
        "test_macro_f1": _r(test, "macro_f1"),
        "test_weighted_f1": _r(test, "weighted_f1"),
        "test_top3": _r(test, "top3"),
        "test_top5": _r(test, "top5"),
        "test_top10": _r(test, "top10"),
        "head_f1": _r(test, "head_f1"),
        "torso_f1": _r(test, "torso_f1"),
        "tail_f1": _r(test, "tail_f1"),
        "notes": notes,
    }
    return row


def write_summary(prefix, rows):
    df = pd.DataFrame([{k: r[k] for k in SUMMARY_COLUMNS} for r in rows],
                      columns=SUMMARY_COLUMNS)
    df.to_csv(os.path.join(HERE, f"{prefix}_summary.csv"), index=False, encoding="utf-8")
    clean = [{k: r[k] for k in SUMMARY_COLUMNS} for r in rows]
    with open(os.path.join(HERE, f"{prefix}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2, default=str)


def write_trials_history(study, prefix):
    rows = []
    for t in study.trials:
        row = {"number": t.number, "state": str(t.state).split(".")[-1],
               "value": t.value, "duration_s": round(
                   (t.datetime_complete - t.datetime_start).total_seconds(), 1)
               if t.datetime_complete and t.datetime_start else ""}
        row.update(t.params)
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(HERE, f"{prefix}_trials_history.csv"),
                              index=False, encoding="utf-8")


def write_report(path, title, sections):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}"
                f" | Companion: DeepLearning_Supplementary_Plan.md\n\n")
        for head, body in sections:
            f.write(f"## {head}\n\n{body}\n\n")
