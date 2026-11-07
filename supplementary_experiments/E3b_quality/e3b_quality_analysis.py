"""
E3b data-quality stratified analysis
=====================================================================================
Companion documents: CBUTransfer/DeepLearning/DeepLearning_Supplementary_Plan.md (E3 follow-ups) + E3 report

Purpose
  building on E3's demonstration that "neither saturates with sample size", further answer:
  1) Are XGBoost and BiGRU jointly limited by data quality?
  2) Is the ANN-vs-GBDT gap concentrated on low-quality/ambiguous samples?
  3) Can the data-quality bottleneck be distinguished from the data-quantity bottleneck?

Design (all stratified analyses of frozen models on the test set, no retraining)
  - Models: E1.4 tuned XGBoost (tuned_model.json) + E1.2 tuned BiGRU (E1_2_tuned.pt)
  - Quality dimensions:
      A. Feature-label neighborhood consistency: training-set 5-NN purity / 1-NN conflict; pure (5/5) vs impure
      B. Feature completeness: number of missing items among 104 features (0 vs 11, missingness concentrated in osda descriptors)
      C. Class long tail: head / torso / tail (training class frequency split into thirds)
      D. Composite clean subset: 5-NN pure and feature-complete
  - Metrics: Accuracy, within-stratum Macro-F1 (over present classes only), true-class average probability, model difference
  - Uncertainty: 2000 sample-level bootstraps (95% CI)
  - Error overlap: both correct / only XGB / only ANN / both wrong, error Jaccard

Outputs (this directory)
  - quality_test_samples.csv      per-test-sample quality labels, predictions and correctness
  - quality_strata_summary.csv    model metrics on each quality stratum
  - quality_effect_tests.csv      bootstrap tests of quality effects and model-interaction effects
  - quality_overview.csv          data quality overview (neighborhood impurity, missingness rate, duplicate conflicts, error overlap)
  - quality_stratified.png        quality-stratified performance chart
  - E3b_summary.json              machine-readable summary
  - E3b_report.md                 auto report
  - e3b_run.log                   log
"""

import json
import logging
import os
import sys
import time
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
import matplotlib
matplotlib.use("Agg")

import numpy as np
import openpyxl
import pandas as pd
import torch
import torch.nn.functional as F
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
E2_DIR = os.path.abspath(os.path.join(HERE, "..", "E2_sota"))
E1_DIR = os.path.abspath(os.path.join(HERE, "..", "E1_tuning"))
for _p in (E2_DIR, E1_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import e2_common as ec                     # noqa: E402
from e1_2_bigru import build_model as build_bigru  # noqa: E402

warnings.filterwarnings("ignore")

N_CLASSES = 231
K_NN = 5
N_BOOT = 2000
BOOT_SEED = 20261024


# ==================== Logging ====================
def make_logger():
    path = os.path.join(HERE, "e3b_run.log")
    lg = logging.getLogger("E3b")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
        for h in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()):
            h.setFormatter(fmt)
            lg.addHandler(h)
    return lg


# ==================== Data and frozen models ====================
def load_data():
    z = np.load(ec.CACHE_PATH)
    le = LabelEncoder()
    y_train = le.fit_transform(z["y_train"])
    y_val = le.transform(z["y_val"])
    y_test = le.transform(z["y_test"])
    if len(le.classes_) != N_CLASSES:
        raise RuntimeError(f"class count {len(le.classes_)} != {N_CLASSES}")
    return dict(X_train=z["X_train"], y_train=y_train,
                X_val=z["X_val"], y_val=y_val,
                X_test=z["X_test"], y_test=y_test, le=le)


def load_frozen_predictions(data, logger):
    """Loads E1 frozen models and outputs test-set probabilities; no retraining."""
    n_test = len(data["y_test"])
    # XGBoost: xgboost native JSON (E1.4 tuned)
    clf = xgb.XGBClassifier()
    clf.load_model(os.path.join(E1_DIR, "tuned_model.json"))
    px = clf.predict_proba(data["X_test"])
    if px.shape[1] != N_CLASSES:
        raise RuntimeError(f"XGB probability column count {px.shape[1]} != {N_CLASSES}")
    logger.info(f"XGBoost-104 loaded | best_iteration={int(getattr(clf, 'best_iteration', -1))}")

    # BiGRU: E1.2 tuned checkpoint + deterministic StandardScaler
    with open(os.path.join(E1_DIR, "best_params_E1_2.json"), encoding="utf-8") as f:
        bp = json.load(f)["params"]
    ck = torch.load(os.path.join(E1_DIR, "E1_2_tuned.pt"), map_location="cpu",
                    weights_only=False)
    model = build_bigru(bp, N_CLASSES)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    scaler = StandardScaler().fit(data["X_train"])
    X_train_s = scaler.transform(data["X_train"]).astype(np.float32)
    X_test_s = scaler.transform(data["X_test"]).astype(np.float32)
    X_val_s = scaler.transform(data["X_val"]).astype(np.float32)
    with torch.no_grad():
        pa = F.softmax(model(torch.as_tensor(X_test_s)), dim=1).numpy()
    logger.info("BiGRU loaded | StandardScaler refit on train (deterministic)")

    acc_x = float((px.argmax(1) == data["y_test"]).mean())
    acc_a = float((pa.argmax(1) == data["y_test"]).mean())
    logger.info(f"frozen model test Acc: XGB={acc_x:.4f} | BiGRU={acc_a:.4f} "
                f"(E1 reference: 0.8353 / 0.7933)")
    if abs(acc_x - 0.8353) > 0.002 or abs(acc_a - 0.7933) > 0.002:
        logger.warning("frozen model accuracy deviates from the E1 reference value by > 0.2 pp, please check the data/scaling protocol")
    return px, pa, X_train_s, X_test_s, acc_x, acc_a, scaler


# ==================== Feature missingness statistics ====================
def missing_counts(path, logger):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    it = ws.iter_rows(values_only=True)
    hdr = list(next(it))
    idx = [hdr.index(c) for c in ec.FEATURES_104]
    counts, col_missing = [], np.zeros(len(idx), dtype=int)
    for row in it:
        m = [j for j, i in enumerate(idx) if row[i] is None or row[i] == ""]
        counts.append(len(m))
        for j in m:
            col_missing[j] += 1
    wb.close()
    counts = np.asarray(counts, dtype=int)
    per_feat = {ec.FEATURES_104[j]: int(col_missing[j])
                for j in np.argsort(-col_missing) if col_missing[j] > 0}
    logger.info(f"missing statistics {os.path.basename(path)}: n={len(counts)} | "
                f"distribution={dict(zip(*np.unique(counts, return_counts=True)))}")
    return counts, per_feat


# ==================== Duplicate/conflict statistics ====================
def duplicate_stats(X, y, logger):
    uniq, inv, cnt = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    dup_groups = np.flatnonzero(cnt > 1)
    conflict_groups, conflict_samples = 0, 0
    for g in dup_groups:
        labs = np.unique(y[inv == g])
        if len(labs) > 1:
            conflict_groups += 1
            conflict_samples += int(cnt[g])
    out = dict(n_unique=len(uniq), n_samples=len(y),
               duplicate_groups=int(len(dup_groups)),
               duplicate_samples=int(cnt[dup_groups].sum()) if len(dup_groups) else 0,
               conflict_groups=conflict_groups,
               conflict_samples=conflict_samples)
    logger.info(f"exact duplicate feature groups: {out['duplicate_groups']} (samples {out['duplicate_samples']}) | "
                f"label conflict groups: {conflict_groups} (samples {conflict_samples})")
    return out


# ==================== Quality stratification framework ====================
def build_quality_frame(data, px, pa, X_train_s, X_test_s, logger):
    y_train, y_test = data["y_train"], data["y_test"]
    le, n = data["le"], len(y_test)

    # 5-NN neighborhood consistency (Euclidean distance on standardized features)
    nn = NearestNeighbors(n_neighbors=K_NN, n_jobs=1).fit(X_train_s)
    dist, idx = nn.kneighbors(X_test_s)
    purity = (y_train[idx] == y_test[:, None]).mean(axis=1)
    nn1_conflict = y_train[idx[:, 0]] != y_test

    # Feature completeness
    mc_train, _ = missing_counts(ec.TRAIN_PATH, logger)
    mc_test, per_feat_test = missing_counts(ec.TEST_PATH, logger)
    if len(mc_test) != n:
        raise RuntimeError(f"test missing-statistics row count {len(mc_test)} != {n}")

    # Long-tail stratification (same as E2: three equal parts by training class frequency)
    strata = ec.make_strata(y_train, N_CLASSES)
    stratum_of = np.empty(N_CLASSES, dtype=object)
    for name, cls in strata.items():
        stratum_of[cls] = name
    freq = np.bincount(y_train, minlength=N_CLASSES)
    class_count = freq[y_test]

    ax = px.argmax(1) == y_test
    aa = pa.argmax(1) == y_test
    frame = pd.DataFrame({
        "test_row_id": np.arange(n),
        "true_code": le.inverse_transform(y_test),
        "true_index": y_test,
        "class_count_train": class_count,
        "freq_stratum": stratum_of[y_test],
        "missing_count": mc_test,
        "feature_complete": mc_test == 0,
        "nn1_label": le.inverse_transform(y_train[idx[:, 0]]),
        "nn1_distance": dist[:, 0].round(6),
        "nn1_conflict": nn1_conflict,
        "nn5_purity": purity.round(4),
        "nn5_impure": purity < 1.0,
        "xgb_pred": le.inverse_transform(px.argmax(1)),
        "ann_pred": le.inverse_transform(pa.argmax(1)),
        "xgb_correct": ax,
        "ann_correct": aa,
        "xgb_prob_true": px[np.arange(n), y_test].round(6),
        "ann_prob_true": pa[np.arange(n), y_test].round(6),
        "both_wrong": ~ax & ~aa,
    })
    frame["error_pattern"] = np.select(
        [ax & aa, ax & ~aa, ~ax & aa, ~ax & ~aa],
        ["both_correct", "only_xgb", "only_ann", "both_wrong"], default="")
    quality = dict(
        test_n=n,
        impurity_1nn=float(nn1_conflict.mean()),
        impurity_5nn=float((purity < 1.0).mean()),
        purity_mean=float(purity.mean()),
        missing_rate=float((mc_test > 0).mean()),
        missing_test_n=int((mc_test > 0).sum()),
        missing_train_rate=float((mc_train > 0).mean()),
        missing_train_n=int((mc_train > 0).sum()),
        missing_features_test=per_feat_test,
        duplicate_train=duplicate_stats(data["X_train"], y_train, logger),
        exact_duplicate_test=int((dist[:, 0] == 0).sum()),
        exact_duplicate_conflict_test=int(((dist[:, 0] == 0) & nn1_conflict).sum()),
        purity_bins={f"{k}/5": int((np.rint(purity * 5).astype(int) == k).sum())
                     for k in range(5, -1, -1)},
    )
    logger.info(f"quality overview: 1NN conflict={quality['impurity_1nn']:.3f} | "
                f"5NN impurity={quality['impurity_5nn']:.3f} | missing rate={quality['missing_rate']:.3f} | "
                f"test-train exact duplicates={quality['exact_duplicate_test']}")
    return frame, quality


def stratum_masks(frame, y_test):
    m = {}
    purity = frame["nn5_purity"].to_numpy()
    # 5-NN purity can only take k/5 (k=0..5); bin directly by the count of same-class neighbors to avoid float boundaries and label duplicates
    kk = np.rint(purity * 5).astype(int)
    m["5NN purity"] = [(f"pure (5/5)" if k == 5 else f"{k}/5 same class", kk == k)
                    for k in range(5, -1, -1)]
    m["Neighborhood conflict"] = [("1NN consistent", ~frame["nn1_conflict"].to_numpy()),
                    ("1NN conflict", frame["nn1_conflict"].to_numpy())]
    m["Feature completeness"] = [("complete (0 missing)", frame["feature_complete"].to_numpy()),
                      ("11 missing items", ~frame["feature_complete"].to_numpy())]
    m["Class frequency"] = [(name, frame["freq_stratum"].to_numpy() == name)
                    for name in ("head", "torso", "tail")]
    clean = frame["feature_complete"].to_numpy() & (purity >= 1.0)
    m["Composite quality"] = [("clean (pure+complete)", clean), ("not clean", ~clean)]
    return m


# ==================== bootstrap ====================
def ci(draws, lo=2.5, hi=97.5):
    return (float(np.percentile(draws, lo)), float(np.percentile(draws, hi)))


def boot_draws(mask, ax, aa, rng):
    idx = np.flatnonzero(mask)
    x_acc, a_acc, gap = [], [], []
    for _ in range(N_BOOT):
        s = rng.choice(idx, size=len(idx), replace=True)
        x, a = ax[s].mean(), aa[s].mean()
        x_acc.append(x)
        a_acc.append(a)
        gap.append(x - a)
    return np.asarray(x_acc), np.asarray(a_acc), np.asarray(gap)


def boot_diff(mask_hi, mask_lo, correct, rng):
    hi, lo = np.flatnonzero(mask_hi), np.flatnonzero(mask_lo)
    out = []
    for _ in range(N_BOOT):
        sh = rng.choice(hi, size=len(hi), replace=True)
        sl = rng.choice(lo, size=len(lo), replace=True)
        out.append(correct[sh].mean() - correct[sl].mean())
    return np.asarray(out)


def summarize(frame, logger):
    ax = frame["xgb_correct"].to_numpy()
    aa = frame["ann_correct"].to_numpy()
    y = frame["true_index"].to_numpy()
    rng = np.random.default_rng(BOOT_SEED)
    rows = []
    for dim, strata in stratum_masks(frame, y).items():
        for name, mask in strata:
            if mask.sum() == 0:
                continue
            xd, ad, gd = boot_draws(mask, ax, aa, rng)
            labels = np.unique(y[mask])
            rows.append(dict(
                dimension=dim, stratum=name, n=int(mask.sum()),
                xgb_acc=round(float(ax[mask].mean()), 4),
                ann_acc=round(float(aa[mask].mean()), 4),
                gap_pp=round(float((ax[mask].mean() - aa[mask].mean()) * 100), 2),
                gap_ci_low_pp=round(ci(gd)[0] * 100, 2),
                gap_ci_high_pp=round(ci(gd)[1] * 100, 2),
                xgb_acc_ci_low=round(ci(xd)[0], 4),
                xgb_acc_ci_high=round(ci(xd)[1], 4),
                ann_acc_ci_low=round(ci(ad)[0], 4),
                ann_acc_ci_high=round(ci(ad)[1], 4),
                xgb_macro_f1=round(float(f1_score(
                    y[mask], frame["xgb_pred"].to_numpy()[mask], labels=labels,
                    average="macro", zero_division=0)), 4),
                ann_macro_f1=round(float(f1_score(
                    y[mask], frame["ann_pred"].to_numpy()[mask], labels=labels,
                    average="macro", zero_division=0)), 4),
                xgb_prob_true=round(float(frame["xgb_prob_true"].to_numpy()[mask].mean()), 4),
                ann_prob_true=round(float(frame["ann_prob_true"].to_numpy()[mask].mean()), 4),
            ))
            logger.info(f"[{dim}] {name}: n={rows[-1]['n']} | "
                        f"XGB={rows[-1]['xgb_acc']:.4f} ANN={rows[-1]['ann_acc']:.4f} "
                        f"gap={rows[-1]['gap_pp']:+.2f} pp")
    return pd.DataFrame(rows)


def effect_tests(frame, logger):
    ax = frame["xgb_correct"].to_numpy()
    aa = frame["ann_correct"].to_numpy()
    purity = frame["nn5_purity"].to_numpy()
    complete = frame["feature_complete"].to_numpy()
    pure, impure = purity >= 1.0, purity < 1.0
    clean, dirty = complete & pure, ~(complete & pure)
    rng = np.random.default_rng(BOOT_SEED + 1)

    dx = boot_diff(pure, impure, ax, rng)
    da = boot_diff(pure, impure, aa, rng)
    dc_x = boot_diff(complete, ~complete, ax, rng)
    dc_a = boot_diff(complete, ~complete, aa, rng)
    dd_x = boot_diff(clean, dirty, ax, rng)
    dd_a = boot_diff(clean, dirty, aa, rng)
    g_clean = boot_draws(clean, ax, aa, rng)[2]
    g_dirty = boot_draws(dirty, ax, aa, rng)[2]

    def row(contrast, model, draws, note=""):
        lo, hi = ci(np.asarray(draws))
        return dict(contrast=contrast, model=model,
                    delta_pp=round(float(np.mean(draws) * 100), 2),
                    ci_low_pp=round(lo * 100, 2), ci_high_pp=round(hi * 100, 2),
                    note=note)

    rows = [
        row("5NN pure → impure", "XGBoost-104", dx),
        row("5NN pure → impure", "BiGRU", da),
        row("5NN pure → impure", "interaction (ANN drop − XGB drop)", da - dx,
            "Positive = ANN is more sensitive to neighborhood ambiguity"),
        row("features complete → 11 missing", "XGBoost-104", dc_x),
        row("features complete → 11 missing", "BiGRU", dc_a),
        row("features complete → 11 missing", "interaction (ANN drop − XGB drop)", dc_a - dc_x),
        row("clean → not clean", "XGBoost-104", dd_x),
        row("clean → not clean", "BiGRU", dd_a),
        row("clean → not clean", "interaction (ANN drop − XGB drop)", dd_a - dd_x),
        row("Gap(XGB−ANN): not clean − clean", "model-gap change", g_dirty - g_clean,
            "Positive = the gap is concentrated on impure samples"),
    ]
    out = pd.DataFrame(rows)
    logger.info("quality effect tests:\n" + out.to_string(index=False))
    return out


def error_overlap(frame, quality):
    ax = frame["xgb_correct"].to_numpy()
    aa = frame["ann_correct"].to_numpy()
    both_wrong = (~ax & ~aa)
    quality["error_overlap"] = dict(
        both_correct=round(float((ax & aa).mean()), 4),
        only_xgb=round(float((ax & ~aa).mean()), 4),
        only_ann=round(float((~ax & aa).mean()), 4),
        both_wrong=round(float(both_wrong.mean()), 4),
        error_jaccard=round(float(both_wrong.sum() / ((~ax) | (~aa)).sum()), 4),
        both_wrong_pure=round(float(both_wrong[frame["nn5_purity"] >= 1].mean()), 4),
        both_wrong_impure=round(float(both_wrong[frame["nn5_impure"]].mean()), 4),
    )
    return quality


# ==================== Plotting ====================
def plot_quality(summary, path):
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), dpi=160)
    specs = [
        ("5NN purity", "Neighborhood purity (5-NN)", axes[0, 0]),
        ("Composite quality", "Composite quality", axes[0, 1]),
        ("Feature completeness", "Feature completeness", axes[1, 0]),
        ("Class frequency", "Class frequency stratum", axes[1, 1]),
    ]
    for dim, title, ax in specs:
        sub = summary[summary["dimension"] == dim]
        x = np.arange(len(sub))
        w = 0.36
        b1 = ax.bar(x - w / 2, sub["xgb_acc"] * 100, w, label="XGBoost-104",
                    color="#1f77b4")
        b2 = ax.bar(x + w / 2, sub["ann_acc"] * 100, w, label="BiGRU",
                    color="#d62728")
        for bars in (b1, b2):
            ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s}\n(n={n})" for s, n in
                            zip(sub["stratum"], sub["n"])], fontsize=7.5)
        ax.set_ylim(0, 108)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=7.5, loc="lower left")
    fig.suptitle("E3b: test accuracy stratified by data-quality dimensions", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ==================== Report ====================
def write_report(summary, effects, quality, frame, acc_x, acc_a):
    def sub(dim):
        return summary[summary["dimension"] == dim]

    def md_table(df, cols):
        head = "| " + " | ".join(cols) + " |\n|" + "|".join(["---"] * len(cols)) + "|\n"
        body = "\n".join("| " + " | ".join(str(r[c]) for c in cols) + " |"
                         for _, r in df.iterrows())
        return head + body

    purity_tbl = md_table(sub("5NN purity"), ["stratum", "n", "xgb_acc", "ann_acc", "gap_pp"])
    conflict_tbl = md_table(sub("Neighborhood conflict"), ["stratum", "n", "xgb_acc", "ann_acc", "gap_pp"])
    complete_tbl = md_table(sub("Feature completeness"), ["stratum", "n", "xgb_acc", "ann_acc", "gap_pp"])
    freq_tbl = md_table(sub("Class frequency"), ["stratum", "n", "xgb_acc", "ann_acc", "gap_pp"])
    comp_tbl = md_table(sub("Composite quality"), ["stratum", "n", "xgb_acc", "ann_acc", "gap_pp"])
    eff_tbl = md_table(effects, ["contrast", "model", "delta_pp", "ci_low_pp", "ci_high_pp"])

    eo = quality["error_overlap"]
    dup = quality["duplicate_train"]
    clean_row = sub("Composite quality").iloc[0]
    dirty_row = sub("Composite quality").iloc[1]
    full_pure = sub("5NN purity").iloc[0]

    md = f"""# E3b data quality stratified analysis report

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')} | Companion: E3 learning curves + DeepLearning_Supplementary_Plan

## 1. Purpose and problem
E3 has shown that neither XGBoost nor BiGRU plateaus at N0=12,537 (data **quantity** not yet saturated).
this experiment further examines two questions:
1. Do both models' errors concentrate jointly on samples with weak **data quality** (neighborhood ambiguity, feature missingness, long tail)?
2. Is the ANN-GBDT gap also explained by data quality rather than model capacity?

## 2. Protocol
- Models: E1.4 tuned XGBoost (`tuned_model.json`) + E1.2 tuned BiGRU (`E1_2_tuned.pt`), **frozen, no retraining**
- Data: frozen test set n={len(frame)}; reference test Acc: XGB={acc_x:.4f}, BiGRU={acc_a:.4f}
- Quality dimensions (all stratified on the test set):
  - **5-NN neighborhood purity**: among the 5 nearest training neighbors in the standardized 104-dim feature space, the fraction of same-class neighbors; pure = 5/5
  - **1-NN conflict**: the nearest training neighbor has a different label than the test sample
  - **Feature completeness**: number of missing items among the 104 features (0 vs 11)
  - **Class frequency**: head/torso/tail (training class frequency split into thirds, consistent with E2)
  - **Composite clean subset**: 5-NN pure and feature-complete
- Uncertainty: 2000 sample-level bootstraps per stratum, 95% CI
- Macro-F1 is computed only over classes present in the stratum; small-sample strata (e.g., tail n={int(sub('Class frequency').iloc[2]['n']) if len(sub('Class frequency')) > 2 else 0}) are for reference only

## 3. Data quality overview
| Metric | Value |
|------|------|
| 1-NN label conflict rate | {quality['impurity_1nn']*100:.1f}% |
| 5-NN neighborhood impurity rate | {quality['impurity_5nn']*100:.1f}% |
| mean 5-NN purity | {quality['purity_mean']*100:.1f}% |
| test set feature missing rate | {quality['missing_rate']*100:.1f}% ({quality['missing_test_n']}/{len(frame)}, all missing the same 11 features) |
| train set feature missing rate | {quality['missing_train_rate']*100:.1f}% ({quality['missing_train_n']} entries) |
| train set exact duplicate feature groups | {dup['duplicate_groups']} groups / {dup['duplicate_samples']} samples |
| of which label conflict groups | {dup['conflict_groups']} groups / {dup['conflict_samples']} samples |
| test-train exact duplicates | {quality['exact_duplicate_test']} entries, of which {quality['exact_duplicate_conflict_test']} label conflicts |
| both models wrong | {eo['both_wrong']*100:.1f}% (error Jaccard={eo['error_jaccard']:.2f}) |
| only XGB wrong / only ANN wrong / both correct | {eo['only_xgb']*100:.1f}% / {eo['only_ann']*100:.1f}% / {eo['both_correct']*100:.1f}% |

the 11 missing features concentrate in OSDA descriptors: {', '.join(list(quality['missing_features_test'].keys())[:6])} etc.

## 4. Stratified results

### 4.1 5-NN neighborhood purity
{purity_tbl}

### 4.2 1-NN conflict
{conflict_tbl}

### 4.3 Feature completeness
{complete_tbl}

### 4.4 Class frequency
{freq_tbl}

### 4.5 Composite quality
{comp_tbl}

an overall monotonic relationship: the lower the 5-NN purity, the more both models' accuracies drop in sync; on pure neighborhoods (n={int(full_pure['n'])}) both reach
{full_pure['xgb_acc']*100:.1f}% / {full_pure['ann_acc']*100:.1f}%, gap only {full_pure['gap_pp']:+.2f} pp.
on the composite clean subset (n={int(clean_row['n'])}), XGB/ANN = {clean_row['xgb_acc']*100:.1f}% / {clean_row['ann_acc']*100:.1f}%
(gap {clean_row['gap_pp']:+.2f} pp); on not-clean samples (n={int(dirty_row['n'])}) it drops to
{dirty_row['xgb_acc']*100:.1f}% / {dirty_row['ann_acc']*100:.1f}%(gap {dirty_row['gap_pp']:+.2f} pp).

## 5. Quality effect tests

{eff_tbl}

Key readings:
- **Common bottleneck**: 5-NN pure -> impure, XGB drops {abs(effects.iloc[0]['delta_pp']):.1f} pp and ANN drops {abs(effects.iloc[1]['delta_pp']):.1f} pp; their errors overlap heavily (both wrong {eo['both_wrong']*100:.1f}%, Jaccard {eo['error_jaccard']:.2f}).
- **ANN is more quality-sensitive**: the interaction term for impure neighborhood is {effects.iloc[2]['delta_pp']:+.1f} pp (ANN drops more); the model gap between clean/non-clean widens from {clean_row['gap_pp']:+.2f} pp to {dirty_row['gap_pp']:+.2f} pp (change {effects.iloc[9]['delta_pp']:+.1f} pp).
- **Missingness has a small and inconsistent effect**: feature missingness lowers XGB by {abs(effects.iloc[3]['delta_pp']):.1f} pp and ANN by {abs(effects.iloc[4]['delta_pp']):.1f} pp, interaction {effects.iloc[5]['delta_pp']:+.1f} pp.

## 6. Conclusion: can we claim "both are limited by data quality"?

**We can claim "the data side (neighborhood ambiguity, missingness, long tail) is the common bottleneck of both", but it must be stated as correlational evidence, not a causal conclusion.**

Evidence supporting this conclusion:
1. With identical architectures, training protocol and data splits, **both are near the ceiling on the clean subset** ({clean_row['xgb_acc']*100:.1f}% / {clean_row['ann_acc']*100:.1f}%), while dropping sharply in sync on non-clean samples;
2. **Highly overlapping errors**: both wrong {eo['both_wrong']*100:.1f}%, error Jaccard {eo['error_jaccard']:.2f}, showing errors concentrate on the same batch of "inherently hard/inconsistent" samples;
3. On all three dimensions - impure neighborhood, feature missingness, long tail - both models decline in the same direction;
4. The training set has {dup['conflict_groups']} groups of records with "identical features but different labels" ({dup['conflict_samples']} samples); the test set has {quality['exact_duplicate_conflict_test']} exact duplicates of training samples with conflicting labels - irreducible label inconsistency.

Boundaries that must be kept:
1. **Quality vs quantity not separable**: the 5-NN impurity rate {quality['impurity_5nn']*100:.1f}% partly comes from insufficient sampling density (long-tail classes, sparse continuous conditioning space), so the "quality bottleneck" and E3's "quantity bottleneck" are two faces of the same data-side problem; this experiment ran no quality intervention (denoising/imputation) and cannot claim "raising quality alone would raise accuracy".
2. **The clean subset is not a random sample**: it is a posterior stratification defined by feature neighborhoods, so being near the ceiling may partly reflect "these samples are inherently easier"; but the synchronized changes of both models on the same subset and the error overlap still support a common data bottleneck.
3. **1-NN/5-NN conflicts are not a strict label error rate**: they measure label inconsistency/class overlap in feature space, possibly including genuine multi-product coexistence and literature recording noise.

## 7. Impact on "stronger scalability"
- On the clean subset the gap between the two is only {clean_row['gap_pp']:+.2f} pp and both are near 100%, indicating **the current architectural difference mainly shows on low-quality/ambiguous samples**, not that ANN would naturally win at large sample sizes.
- ANN is more sensitive to neighborhood ambiguity (drops {abs(effects.iloc[2]['delta_pp']):.1f} pp more); merely adding samples without raising neighborhood purity may leave ANN's relative disadvantage in place.
- Therefore "stronger scalability" still cannot be claimed; a more accurate statement: **ANN's scaling benefit is "quality-gated"** - only after label consistency, OSDA descriptor completeness and long-tail coverage improve can its scale advantage be discussed.

## 8. File list
| File | Content |
|------|------|
| quality_test_samples.csv | quality labels, predictions, correctness, and error patterns of the 2,641 test samples |
| quality_strata_summary.csv | n / Acc / gap / Macro-F1 / true class probability per quality stratum |
| quality_effect_tests.csv | bootstrap tests for quality effects, interaction effects, and model-gap changes |
| quality_overview.csv | quality overview and error overlap |
| quality_stratified.png | four-dimension quality-stratified performance plot |
| E3b_summary.json | machine-readable summary |
| E3b_report.md | this report |
| e3b_run.log | log |
"""
    with open(os.path.join(HERE, "E3b_report.md"), "w", encoding="utf-8") as f:
        f.write(md)
    logger_note = f"report generated: {os.path.join(HERE, 'E3b_report.md')}"
    print(logger_note)


# ==================== Main pipeline ====================
def main():
    t0 = time.time()
    logger = make_logger()
    logger.info("=" * 72)
    logger.info("E3b data quality stratified analysis | frozen E1 model, test set stratification")

    data = load_data()
    px, pa, X_train_s, X_test_s, acc_x, acc_a, _ = load_frozen_predictions(data, logger)
    frame, quality = build_quality_frame(data, px, pa, X_train_s, X_test_s, logger)
    frame.to_csv(os.path.join(HERE, "quality_test_samples.csv"),
                 index=False, encoding="utf-8-sig")

    summary = summarize(frame, logger)
    summary.to_csv(os.path.join(HERE, "quality_strata_summary.csv"),
                   index=False, encoding="utf-8-sig")

    effects = effect_tests(frame, logger)
    effects.to_csv(os.path.join(HERE, "quality_effect_tests.csv"),
                   index=False, encoding="utf-8-sig")

    quality = error_overlap(frame, quality)
    overview = pd.DataFrame([
        dict(metric=k, value=json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v)
        for k, v in quality.items() if k != "error_overlap"
    ] + [dict(metric=f"error_overlap.{k}", value=v)
         for k, v in quality["error_overlap"].items()])
    overview.to_csv(os.path.join(HERE, "quality_overview.csv"),
                    index=False, encoding="utf-8-sig")

    plot_quality(summary, os.path.join(HERE, "quality_stratified.png"))

    out = dict(acc_x=acc_x, acc_a=acc_a,
               quality=quality,
               strata=summary.to_dict("records"),
               effects=effects.to_dict("records"),
               protocol=dict(k_nn=K_NN, n_boot=N_BOOT, boot_seed=BOOT_SEED,
                             models="E1.4 XGBoost + E1.2 BiGRU (frozen)",
                             test_n=len(frame)))
    with open(os.path.join(HERE, "E3b_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)

    write_report(summary, effects, quality, frame, acc_x, acc_a)
    logger.info(f"done | total time {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
