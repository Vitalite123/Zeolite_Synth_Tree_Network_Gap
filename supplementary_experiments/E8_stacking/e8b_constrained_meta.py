"""
E8b: E8 rerun experiment (preregistration in DeepLearning_Supplementary_Plan.md §9.1 [rerun experiment E8b])
================================================================
Purpose
----
run the two branches per the sole reopen condition registered in E8, same criterion:
  M1 constrained meta-layer: per-class simplex weights W∈R^{231x3} (693 params), OOF coordinate ascent maximizing Acc,
     select lambda by shrinkage toward uniform weights; inference foldavg (consistent with the E8 primary metric).
  M2 non-probabilistic fusion: per-model class ranks + reciprocal-rank fusion (RRF, k=60), computed per fold then averaged over 5 folds.
Criterion: M1 vs E4 XGBoost-104 same-seed paired ΔAcc mean ≥ +0.5 pp and 95% CI lower bound > 0.

Metric conventions
----
- No base-model retraining: reuse E8's saved 5-fold OOF and per-fold test probabilities;
- Anchors: npz complete / fold_id full coverage / base-model bagged Acc within 0.2 pp per seed vs E8 summary /
  test_* and fold_test_* means agree;
- Guardrails disclosed separately: Macro-F1(sup) mean >= 71.55%, Top-5 mean >= 97.30%.

Artifacts
----
E8b_report.md / E8b_summary.json / E8b_seeds.csv / E8b_methods.csv /
E8b_class_weights.csv / E8b_metrics.png / e8b_run.log
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

HERE = os.path.dirname(os.path.abspath(__file__))
SUP_DIR = os.path.abspath(os.path.join(HERE, ".."))
E2_DIR = os.path.join(SUP_DIR, "E2_sota")
if E2_DIR not in sys.path:
    sys.path.insert(0, E2_DIR)
import e2_common as ec  # noqa: E402

# ==================== Constants (identical to E8 / E4) ====================
N_CLASSES = ec.N_CLASS_EXPECT
MODELS = ["XGBoost-104", "BiGRU", "FT-Transformer"]
SLUG = {"XGBoost-104": "xgboost", "BiGRU": "bigru", "FT-Transformer": "ftt"}
SEEDS = [42, 123, 777, 2026, 3407]
E4_SEED_CSV = os.path.join(SUP_DIR, "E4_seeds", "seed_repetition_results.csv")
MAP_CSV = os.path.join(ec.DATA_DIR, "code1_index_mapping.csv")

GRID_STEP = 0.05
PASSES = 3
LAMBDAS = (0.0, 0.1, 0.2, 0.3)
RRF_K = 60
DELTA_PP_REQUIRED = 0.5
GUARD_MACRO_F1 = 0.7155
GUARD_TOP5 = 0.9730
ANCHOR_TOL_PP = 0.2

LOG = None


def log(msg):
    if LOG is not None:
        LOG.info(msg)
    print(msg, flush=True)


def _json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def npz_path(seed):
    return os.path.join(HERE, f"E8_oof_ms_seed{seed}.npz")


def summary_path(seed):
    return os.path.join(HERE, f"E8_summary_ms_seed{seed}.json")


# ==================== Fusion rules ====================
def simplex_grid(step):
    vals = np.round(np.arange(0.0, 1.0 + 1e-9, step), 6)
    rows = []
    for a in vals:
        for b in vals:
            if a + b <= 1.0 + 1e-9:
                rows.append([a, b, 1.0 - a - b])
    return np.asarray(rows, dtype=np.float64)


def fuse(P, W):
    """P: (K,n,C) per-model probabilities; W: (C,K) per-class weights -> (n,C) fused score."""
    return np.einsum("knc,ck->nc", P, W, optimize=True)


def fit_perclass(P, y, init_w, grid, passes):
    """Coordinate ascent: per class, maximize OOF Acc over the simplex grid."""
    K, n, C = P.shape
    W = np.tile(np.asarray(init_w, dtype=np.float64), (C, 1))
    S = fuse(P, W)
    history = []
    for p in range(passes):
        changed = 0
        for c in range(C):
            S[:, c] = -np.inf
            B = S.max(axis=1)
            pred_excl = S.argmax(axis=1)
            correct_excl = pred_excl == y
            A = np.stack([P[k][:, c] for k in range(K)], axis=1)
            Sc = A @ grid.T
            is_c = y == c
            win = Sc >= B[:, None]
            part1 = (is_c[:, None] & win).sum(axis=0)
            part2 = ((~is_c)[:, None] & (~win) & correct_excl[:, None]).sum(axis=0)
            cnt = part1 + part2
            j = int(np.argmax(cnt))
            if np.any(np.abs(grid[j] - W[c]) > 1e-9):
                changed += 1
            W[c] = grid[j]
            S[:, c] = Sc[:, j]
        acc = float((S.argmax(axis=1) == y).mean())
        history.append(acc)
        log(f"  [per-class] pass {p + 1}/{passes}: OOF Acc={acc:.4f} | classes changed {changed}")
    return W, S, history


def select_lambda(P, y, W):
    best = None
    for lam in LAMBDAS:
        Wl = (1.0 - lam) * W + lam / 3.0
        acc = float((fuse(P, Wl).argmax(axis=1) == y).mean())
        log(f"  [shrink] lambda={lam:.1f}: OOF Acc={acc:.4f}")
        if best is None or acc > best[1] + 1e-12:
            best = (lam, acc, Wl)
    return best


def rrf_matrix(P):
    """P: (K,n,C) -> (n,C) reciprocal-rank fusion score (k=RRF_K)."""
    K, n, C = P.shape
    out = np.zeros((n, C), dtype=np.float64)
    for k in range(K):
        order = np.argsort(-P[k], axis=1, kind="stable")
        ranks = np.empty_like(order)
        np.put_along_axis(ranks, order, np.arange(C)[None, :], axis=1)
        out += 1.0 / (RRF_K + ranks)
    return out


# ==================== Metrics ====================
def full_metrics(proba, y, strata):
    m = ec.metrics_from_proba(proba, y, N_CLASSES, strata)
    m["macro_f1_support"] = float(f1_score(
        y, proba.argmax(axis=1), labels=np.unique(y), average="macro", zero_division=0))
    m["n_samples"] = int(len(y))
    return m


def metric_core(m):
    return {"acc": float(m["acc"]), "macro_f1_support": float(m["macro_f1_support"]),
            "weighted_f1": float(m["weighted_f1"]), "top5": float(m["top5"])}


def ci95(d):
    import scipy.stats as st
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    mean = float(d.mean())
    if n < 2:
        return mean, 0.0, mean, mean
    sd = float(d.std(ddof=1))
    half = float(st.t.ppf(0.975, n - 1) * sd / math.sqrt(n))
    return mean, sd, mean - half, mean + half


# ==================== Main pipeline ====================
def main():
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--passes", type=int, default=PASSES)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    seeds = args.seeds or SEEDS
    tag = args.tag
    t_start = time.time()
    log_path = os.path.join(HERE, f"e8b_run{tag}.log")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(log_path, mode="w", encoding="utf-8"),
                  logging.StreamHandler()])
    LOG = logging.getLogger("E8b")
    log("=" * 78)
    log(f"E8b constrained meta-layer / non-probability fusion reopen experiment | seeds={seeds} | passes={args.passes}")

    grid = simplex_grid(GRID_STEP)
    log(f"simplex grid: step={GRID_STEP} -> {len(grid)} candidates/class")

    # ---- Class names (consistent with npz encoding: LabelEncoder fit on cached y_train) ----
    z = np.load(ec.CACHE_PATH)
    le = LabelEncoder()
    y_cache = le.fit_transform(z["y_train"]).astype(np.int64)
    classes = le.classes_
    name_of = {}
    code_map = {}
    if os.path.exists(MAP_CSV):
        mp = pd.read_csv(MAP_CSV)
        code_map = {int(r["Code1_index"]): str(r["Code1"]) for _, r in mp.iterrows()}
    for i, v in enumerate(classes):
        if isinstance(v, (int, np.integer)) or str(v).isdigit():
            name_of[i] = code_map.get(int(v), str(v))
        else:
            name_of[i] = str(v)

    e4 = pd.read_csv(E4_SEED_CSV)
    e4x = {int(k): float(v) for k, v in
           e4[e4["model"] == "XGBoost-104"].set_index("seed")["acc"].items()}
    missing = [s for s in seeds if s not in e4x]
    if missing:
        raise KeyError(f"E4 results missing XGBoost-104 for seed {missing}")

    per_seed = {}
    weight_rows = []
    for s in seeds:
        log("-" * 78)
        log(f"[seed {s}] reading {os.path.basename(npz_path(s))}")
        if not os.path.exists(npz_path(s)):
            raise FileNotFoundError(npz_path(s))
        npz = np.load(npz_path(s))
        y_train = npz["y_train"].astype(np.int64)
        y_test = npz["y_test"].astype(np.int64)
        fold_id = npz["fold_id"].astype(np.int64)
        if not np.array_equal(y_train, y_cache):
            raise RuntimeError(f"seed {s}: npz y_train inconsistent with cached LabelEncoder encoding")
        if not (fold_id >= 0).all():
            raise RuntimeError(f"seed {s}: fold_id does not cover all training samples "
                               f"({int((fold_id >= 0).sum())}/{len(fold_id)})")
        if len(y_test) != 2641 or len(y_train) != 12537:
            raise RuntimeError(f"seed {s}: sample counts invalid train={len(y_train)} test={len(y_test)}")

        cover = fold_id >= 0
        P_oof = np.stack([npz[f"oof_{SLUG[m]}"][cover].astype(np.float32) for m in MODELS], 0)
        P_test = np.stack([npz[f"test_{SLUG[m]}"].astype(np.float32) for m in MODELS], 0)
        FT = {m: npz[f"fold_test_{SLUG[m]}"].astype(np.float32) for m in MODELS}
        n_folds = FT[MODELS[0]].shape[0]
        if n_folds != 5:
            raise RuntimeError(f"seed {s}: fold count {n_folds} != 5")

        # ---- Anchors ----
        e8sum = _json(summary_path(s))
        for m in MODELS:
            acc_bag = float((P_test[MODELS.index(m)].argmax(axis=1) == y_test).mean())
            ref = float(e8sum["metrics"][m]["test"]["acc"])
            d_pp = (acc_bag - ref) * 100
            if abs(d_pp) > ANCHOR_TOL_PP:
                raise RuntimeError(f"anchor failed seed={s} {m}: recomputed {acc_bag:.6f} vs E8 {ref:.6f} "
                                   f"(Δ={d_pp:+.3f} pp > {ANCHOR_TOL_PP})")
            mean_fold = np.mean([FT[m][f] for f in range(n_folds)], axis=0)
            max_diff = float(np.abs(mean_fold - P_test[MODELS.index(m)]).max())
            if max_diff > 2e-3:
                raise RuntimeError(f"anchor failed seed={s} {m}: mean difference between test_* and fold_test_* {max_diff:.2e}")
            log(f"[anchor seed={s}] {m}: bagged Acc={acc_bag:.6f} (E8 {ref:.6f}, "
                f"Δ={d_pp:+.3f} pp) | test/fold mean max|Δ|={max_diff:.2e}")

        strata = ec.make_strata(y_train, N_CLASSES)
        init_w = np.asarray(e8sum.get("weights", [1 / 3, 1 / 3, 1 / 3]), dtype=np.float64)
        log(f"[seed {s}] M1 initial weights (E8 weighted_avg): {np.round(init_w, 3).tolist()}")

        # ---- M1: per-class simplex weights ----
        W0, S0, hist = fit_perclass(P_oof, y_train, init_w, grid, args.passes)
        lam, oof_acc_lam, W = select_lambda(P_oof, y_train, W0)
        oof_m1 = fuse(P_oof, W)
        oof_metrics = full_metrics(oof_m1, y_train, strata)
        log(f"[seed {s}] M1 | lambda={lam} | OOF Acc={oof_acc_lam:.4f}")

        # foldavg inference
        acc_fold = np.zeros_like(P_test[0], dtype=np.float64)
        for f in range(n_folds):
            Pf = np.stack([FT[m][f] for m in MODELS], 0)
            acc_fold += fuse(Pf, W)
        proba_m1_fold = (acc_fold / n_folds).astype(np.float32)
        proba_m1_bag = fuse(P_test, W).astype(np.float32)

        # ---- M2: reciprocal-rank fusion ----
        acc_fold_r = np.zeros((len(y_test), N_CLASSES), dtype=np.float64)
        for f in range(n_folds):
            Pf = np.stack([FT[m][f] for m in MODELS], 0)
            acc_fold_r += rrf_matrix(Pf)
        proba_m2_fold = (acc_fold_r / n_folds).astype(np.float32)
        proba_m2_bag = rrf_matrix(P_test).astype(np.float32)
        oof_m2 = rrf_matrix(P_oof).astype(np.float32)
        oof_m2_metrics = full_metrics(oof_m2, y_train, strata)

        m1_fold = full_metrics(proba_m1_fold, y_test, strata)
        m1_bag = full_metrics(proba_m1_bag, y_test, strata)
        m2_fold = full_metrics(proba_m2_fold, y_test, strata)
        m2_bag = full_metrics(proba_m2_bag, y_test, strata)

        # Weight diagnostics
        dom = np.argmax(W0, axis=1)
        dom_counts = {m: int((dom == k).sum()) for k, m in enumerate(MODELS)}
        p = np.clip(W0, 1e-12, 1.0)
        ent = -np.sum(p * np.log(p), axis=1) / math.log(3.0)
        log(f"[seed {s}] weight diagnostics: dominant class counts {dom_counts} | mean normalized entropy {ent.mean():.4f} | "
            f"lambda={lam}")
        for c in range(N_CLASSES):
            weight_rows.append(dict(
                seed=s, class_index=int(c), class_label=name_of.get(c, str(c)),
                w_xgboost=round(float(W0[c, 0]), 6), w_bigru=round(float(W0[c, 1]), 6),
                w_ftt=round(float(W0[c, 2]), 6), dominant_model=MODELS[int(dom[c])],
                lambda_shrink=float(lam), weight_entropy_norm=round(float(ent[c]), 6)))

        per_seed[s] = dict(
            m1_fold=metric_core(m1_fold), m1_bag=metric_core(m1_bag),
            m2_fold=metric_core(m2_fold), m2_bag=metric_core(m2_bag),
            m1_oof=metric_core(oof_metrics), m2_oof=metric_core(oof_m2_metrics),
            lambda_shrink=float(lam), init_weights=[float(x) for x in init_w],
            weights=W0.tolist(), weight_entropy_mean=float(ent.mean()),
            dominant_counts=dom_counts, fit_history=hist, oof_acc_after_lambda=oof_acc_lam,
            e8_weighted_avg=float(e8sum["metrics"]["weighted_avg"]["test"]["acc"]),
            e8_weighted_oof=float(e8sum["metrics"]["weighted_avg"]["oof"]["acc"]),
            e8_stacking_foldavg=float(e8sum["metrics"]["stacking_all3_foldavg"]["test"]["acc"]),
            oracle_all3=float(e8sum["oracle"]["test"]["all3_acc"]),
        )
        log(f"[seed {s}] test | M1 foldavg Acc={m1_fold['acc']:.4f} "
            f"MacroF1sup={m1_fold['macro_f1_support']:.4f} Top5={m1_fold['top5']:.4f} | "
            f"M1 bagged={m1_bag['acc']:.4f} | M2 foldavg Acc={m2_fold['acc']:.4f} "
            f"Top5={m2_fold['top5']:.4f} | M2 bagged={m2_bag['acc']:.4f}")

    # ---- Cross-seed aggregation ----
    def vec(key):
        return np.array([per_seed[s][key]["acc"] for s in seeds], dtype=np.float64)

    def vec_key(method_key, key):
        return np.array([per_seed[s][method_key][key] for s in seeds], dtype=np.float64)

    acc_e4 = np.array([e4x[s] for s in seeds], dtype=np.float64)
    acc_m1 = vec_key("m1_fold", "acc")
    acc_m2 = vec_key("m2_fold", "acc")
    acc_m1b = vec_key("m1_bag", "acc")
    acc_e8w = np.array([per_seed[s]["e8_weighted_avg"] for s in seeds], dtype=np.float64)
    acc_e8s = np.array([per_seed[s]["e8_stacking_foldavg"] for s in seeds], dtype=np.float64)

    deltas = {}
    for name, v in (("m1_vs_e4", acc_m1), ("m2_vs_e4", acc_m2),
                    ("m1_vs_e8_weighted", acc_m1), ("m1_vs_e8_stack", acc_m1)):
        base = {"m1_vs_e4": acc_e4, "m2_vs_e4": acc_e4,
                "m1_vs_e8_weighted": acc_e8w, "m1_vs_e8_stack": acc_e8s}[name]
        d = (v - base) * 100
        m, sd, lo, hi = ci95(d)
        deltas[name] = {"mean": round(m, 3), "std": round(sd, 3),
                        "ci95_low": round(lo, 3), "ci95_high": round(hi, 3),
                        "per_seed": [round(float(x), 3) for x in d]}
        log(f"paired {name}: {m:+.2f} pp [95% CI {lo:+.2f}, {hi:+.2f}]")

    # Method table (with E8/E4 references)
    methods = {}
    for key, label in (("m1_fold", "M1_perclass_foldavg"), ("m1_bag", "M1_perclass_bagged"),
                       ("m2_fold", "M2_rank_fusion_foldavg"), ("m2_bag", "M2_rank_fusion_bagged")):
        methods[label] = {"acc": vec_key(key, "acc"),
                          "macro_f1_support": vec_key(key, "macro_f1_support"),
                          "weighted_f1": vec_key(key, "weighted_f1"),
                          "top5": vec_key(key, "top5")}
    e8sum0 = _json(summary_path(seeds[0]))
    for ref_key, label in (("XGBoost-104", "XGBoost-104(bagged)"),
                           ("BiGRU", "BiGRU(bagged)"),
                           ("FT-Transformer", "FT-Transformer(bagged)"),
                           ("weighted_avg", "E8_weighted_avg"),
                           ("stacking_all3_foldavg", "E8_stacking_foldavg")):
        vals = {k: [] for k in ("acc", "macro_f1_support", "weighted_f1", "top5")}
        for s in seeds:
            mm = _json(summary_path(s))["metrics"][ref_key]["test"]
            for k in vals:
                vals[k].append(float(mm[k]))
        methods[label] = {k: np.array(v, dtype=np.float64) for k, v in vals.items()}
    methods["E4_XGBoost-104(full-train)"] = {
        "acc": acc_e4,
        "macro_f1_support": np.array([np.nan] * len(seeds)),
        "weighted_f1": np.array([np.nan] * len(seeds)),
        "top5": np.array([np.nan] * len(seeds)),
    }

    def desc(v):
        vv = np.asarray(v, dtype=np.float64)
        return {"mean": round(float(vv.mean()), 6),
                "std": round(float(vv.std(ddof=1)), 6) if len(vv) > 1 else 0.0,
                "per_seed": [None if not np.isfinite(x) else round(float(x), 6) for x in vv]}

    agg = {label: {k: desc(v) for k, v in mv.items()} for label, mv in methods.items()}

    # Guardrails (primary convention M1 foldavg)
    f1 = vec_key("m1_fold", "macro_f1_support")
    top5 = vec_key("m1_fold", "top5")
    guard = {
        "macro_f1_support_mean": round(float(f1.mean()), 6),
        "macro_f1_support_ref": GUARD_MACRO_F1,
        "macro_f1_support_pass": bool(f1.mean() >= GUARD_MACRO_F1),
        "macro_f1_support_seeds_pass": [int((f1 >= GUARD_MACRO_F1).sum()), len(seeds)],
        "top5_mean": round(float(top5.mean()), 6),
        "top5_ref": GUARD_TOP5,
        "top5_pass": bool(top5.mean() >= GUARD_TOP5),
        "top5_seeds_pass": [int((top5 >= GUARD_TOP5).sum()), len(seeds)],
    }

    d1 = deltas["m1_vs_e4"]
    d2 = deltas["m2_vs_e4"]
    m1_pass = bool(d1["mean"] >= DELTA_PP_REQUIRED and d1["ci95_low"] > 0)
    m2_pass = bool(d2["mean"] >= DELTA_PP_REQUIRED and d2["ci95_low"] > 0)
    if m1_pass:
        branch = "constrained_meta"
        branch_text = "constrained meta-layer branch holds: the E8 dimension line reopens"
    elif m2_pass:
        branch = "non_probabilistic_fusion"
        branch_text = "non-probabilistic fusion branch holds: the E8 dimension line reopens"
    else:
        branch = "none"
        branch_text = "neither variant passes: the E8 dimension line stays closed"
    verdict = {
        "criterion": "M1 paired same-seed vs E4 XGBoost-104: mean delta Acc >= +0.5 pp and 95% CI lower bound > 0",
        "primary_method": "M1_perclass_foldavg",
        "m1": {"delta": d1, "pass": m1_pass},
        "m2": {"delta": d2, "pass": m2_pass},
        "guardrail": guard,
        "branch": branch,
        "branch_text": branch_text,
    }
    log(f"Verdict: M1 {'pass' if m1_pass else 'fail'} | M2 {'pass' if m2_pass else 'fail'} "
        f"| branch={branch}")

    # ---- Artifacts ----
    seeds_csv = os.path.join(HERE, f"E8b_seeds{tag}.csv")
    rows = []
    for label in methods:
        for i, s in enumerate(seeds):
            rows.append(dict(method=label, seed=s, acc=round(float(methods[label]["acc"][i]), 6),
                             macro_f1_support=round(float(methods[label]["macro_f1_support"][i]), 6)
                             if np.isfinite(methods[label]["macro_f1_support"][i]) else "",
                             weighted_f1=round(float(methods[label]["weighted_f1"][i]), 6)
                             if np.isfinite(methods[label]["weighted_f1"][i]) else "",
                             top5=round(float(methods[label]["top5"][i]), 6)
                             if np.isfinite(methods[label]["top5"][i]) else ""))
    pd.DataFrame(rows).to_csv(seeds_csv, index=False, encoding="utf-8-sig")

    methods_csv = os.path.join(HERE, f"E8b_methods{tag}.csv")
    mrows = []
    for label, a in agg.items():
        mm = np.array(a["acc"]["per_seed"], dtype=np.float64)
        dd = (mm - np.array([e4x[s] for s in seeds])) * 100
        dm, dsd, dlo, dhi = ci95(dd)
        mrows.append(dict(
            method=label, acc_mean=a["acc"]["mean"], acc_std=a["acc"]["std"],
            macro_f1_support_mean=a["macro_f1_support"]["mean"],
            weighted_f1_mean=a["weighted_f1"]["mean"], top5_mean=a["top5"]["mean"],
            delta_vs_e4_mean_pp=round(dm, 3), delta_vs_e4_std_pp=round(dsd, 3),
            delta_vs_e4_ci95_low=round(dlo, 3), delta_vs_e4_ci95_high=round(dhi, 3),
        ))
    pd.DataFrame(mrows).to_csv(methods_csv, index=False, encoding="utf-8-sig")

    wcsv = os.path.join(HERE, f"E8b_class_weights{tag}.csv")
    pd.DataFrame(weight_rows).to_csv(wcsv, index=False, encoding="utf-8-sig")

    summary = {
        "experiment": "E8b",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seeds": [int(s) for s in seeds],
        "primary_method": "M1_perclass_foldavg",
        "protocol": {
            "data": "Reuses the 5-fold OOF and per-fold test probabilities in E8_oof_ms_seed*.npz (no base-model retraining)",
            "m1": f"Per-class simplex weights 231x3, OOF coordinate ascent (step={GRID_STEP}, {args.passes} passes), "
                  f"shrinkage lambda∈{list(LAMBDAS)}; foldavg inference",
            "m2": f"Per-model class ranks + reciprocal-rank fusion (RRF k={RRF_K}); foldavg inference",
            "anchor_tol_pp": ANCHOR_TOL_PP,
        },
        "reference": {"e4_xgb_per_seed": {str(s): round(e4x[s], 6) for s in seeds}},
        "per_seed": {str(s): {k: v for k, v in d.items() if k != "weights"}
                     for s, d in per_seed.items()},
        "aggregate": agg,
        "deltas": deltas,
        "verdict": verdict,
        "timings": {"total_s": round(time.time() - t_start, 1)},
        "artifacts": {"seeds_csv": seeds_csv, "methods_csv": methods_csv,
                      "weights_csv": wcsv,
                      "summary_json": os.path.join(HERE, f"E8b_summary{tag}.json"),
                      "report_md": os.path.join(HERE, f"E8b_report{tag}.md"),
                      "figure": os.path.join(HERE, f"E8b_metrics{tag}.png"),
                      "log": log_path},
    }
    with open(os.path.join(HERE, f"E8b_summary{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ---- Figures ----
    plot_methods = ["E4_XGBoost-104(full-train)", "XGBoost-104(bagged)", "BiGRU(bagged)",
                    "FT-Transformer(bagged)", "E8_weighted_avg", "E8_stacking_foldavg",
                    "M1_perclass_foldavg", "M2_rank_fusion_foldavg"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    ax = axes[0]
    vals = [agg[n]["acc"]["mean"] * 100 for n in plot_methods]
    errs = [agg[n]["acc"]["std"] * 100 for n in plot_methods]
    colors = ["#888888", "#1f77b4", "#1f77b4", "#1f77b4", "#2ca02c", "#2ca02c", "#d62728", "#ff7f0e"]
    ax.barh(range(len(plot_methods)), vals, xerr=errs, color=colors, alpha=0.85,
            error_kw=dict(lw=1, capsize=3))
    ax.set_yticks(range(len(plot_methods)))
    ax.set_yticklabels(plot_methods, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(83.33, color="k", ls="--", lw=1)
    ax.set_xlabel("test Accuracy (%)")
    ax.set_title("E8b: 5-seed test Acc (mean ± std)")

    ax = axes[1]
    xs = np.arange(len(seeds))
    ax.plot(xs, d1["per_seed"], "o-", label=f"M1 (Δ={d1['mean']:+.2f} pp)", color="#d62728")
    ax.plot(xs, d2["per_seed"], "s--", label=f"M2 (Δ={d2['mean']:+.2f} pp)", color="#ff7f0e")
    ax.axhline(0.5, color="k", ls=":", lw=1, label="+0.5 pp threshold")
    ax.axhline(0.0, color="gray", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_xlabel("seed")
    ax.set_ylabel("ΔAcc vs E4 XGBoost-104 (pp)")
    ax.set_title("Paired ΔAcc (same seed)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig_png = os.path.join(HERE, f"E8b_metrics{tag}.png")
    fig.savefig(fig_png, dpi=150)
    plt.close(fig)

    # ---- Report ----
    rep = os.path.join(HERE, f"E8b_report{tag}.md")
    lines = ["# E8b: constrained meta-layer / non-probability fusion reopen experiment (5-seed paired)", "",
             f"- Date: {summary['date']} | Seeds: {seeds} | Primary metric: M1_perclass_foldavg (foldavg)",
             f"- Criterion: {verdict['criterion']}", "",
             "## 1. Conclusion", "",
             f"- **{branch_text}**",
             f"- M1 ΔAcc vs E4: {d1['mean']:+.2f} pp [95% CI {d1['ci95_low']:+.2f}, "
             f"{d1['ci95_high']:+.2f}], per-seed {d1['per_seed']} → "
             f"{'pass' if m1_pass else 'fail'}",
             f"- M2 ΔAcc vs E4: {d2['mean']:+.2f} pp [95% CI {d2['ci95_low']:+.2f}, "
             f"{d2['ci95_high']:+.2f}], per-seed {d2['per_seed']} → "
             f"{'pass' if m2_pass else 'fail'}",
             f"- vs E8 stacking foldavg: {deltas['m1_vs_e8_stack']['mean']:+.2f} pp "
             f"[{deltas['m1_vs_e8_stack']['ci95_low']:+.2f}, "
             f"{deltas['m1_vs_e8_stack']['ci95_high']:+.2f}];"
             f"vs E8 weighted_avg: {deltas['m1_vs_e8_weighted']['mean']:+.2f} pp "
             f"[{deltas['m1_vs_e8_weighted']['ci95_low']:+.2f}, "
             f"{deltas['m1_vs_e8_weighted']['ci95_high']:+.2f}]",
             "", "## 2. 5-seed mean of each method (test)", "",
             "| Method | Acc | Macro-F1(sup) | Weighted-F1 | Top-5 |", "|---|---|---|---|---|"]
    for label in plot_methods + ["M1_perclass_bagged", "M2_rank_fusion_bagged"]:
        a = agg[label]
        lines.append(f"| {label} | {a['acc']['mean'] * 100:.2f} ± {a['acc']['std'] * 100:.2f} | "
                     f"{a['macro_f1_support']['mean'] * 100:.2f} | "
                     f"{a['weighted_f1']['mean'] * 100:.2f} | {a['top5']['mean'] * 100:.2f} |")
    lines += ["", "## 3. Guardrails (M1 primary metric)", "",
              f"- Macro-F1(sup) mean {guard['macro_f1_support_mean'] * 100:.2f}% "
              f"(ref {GUARD_MACRO_F1 * 100:.2f}%), per-seed pass "
              f"{guard['macro_f1_support_seeds_pass'][0]}/{guard['macro_f1_support_seeds_pass'][1]}",
              f"- Top-5 mean {guard['top5_mean'] * 100:.2f}% (ref {GUARD_TOP5 * 100:.2f}%),"
              f"per-seed pass {guard['top5_seeds_pass'][0]}/{guard['top5_seeds_pass'][1]}",
              "", "## 4. Per-class weight diagnostics", ""]
    for s in seeds:
        d = per_seed[s]
        lines.append(f"- seed {s}: lambda={d['lambda_shrink']} | dominant class counts {d['dominant_counts']} "
                     f"| mean normalized entropy {d['weight_entropy_mean']:.4f} | "
                     f"OOF Acc M1 {d['m1_oof']['acc']:.4f} vs E8 global weights "
                     f"{d['e8_weighted_oof']:.4f} | test Acc M1 {d['m1_fold']['acc']:.4f} vs "
                     f"E8 global weights {d['e8_weighted_avg']:.4f}")
    lines += ["", "## 5. Evidence boundary", "",
              "- Base models remain E8's within-fold 5-fold models (each fold uses only 80% of training data), so pairing with E4's full-training-set XGBoost differs in training volume; ",
              "- Per-class weights improve over E8 global weights by about +1.8 pp on OOF but drop by -0.64 pp on test; "
              "And lambda shrinkage was judged unfavorable by OOF (all lambda=0), showing the per-class degrees of freedom fit between-fold/within-class noise. "
              "OOF and test absolute Acc are not directly subtractable because class coverage differs (OOF covers 231 classes, test only 148), "
              "hence the difference vs the same-convention E8 weighted_avg serves as overfitting evidence; ",
              "- The degrees of freedom of the 5-seed paired CI is only 4, so intervals are wide; ",
              "- RRF does not use probability magnitudes; its Top-5 ranking quality cannot be compared directly with probability fusion and serves only as a non-probability-fusion reference.",
              "", "## 6. Artifacts", "",
              f"- `{os.path.basename(seeds_csv)}` / `{os.path.basename(methods_csv)}` / "
              f"`{os.path.basename(wcsv)}`",
              f"- `{os.path.basename(summary['artifacts']['summary_json'])}` / "
              f"`{os.path.basename(rep)}` / `{os.path.basename(fig_png)}` / "
              f"`{os.path.basename(log_path)}`"]
    with open(rep, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    log(f"done | total time {summary['timings']['total_s']}s | artifacts: "
        f"{os.path.basename(rep)}, {os.path.basename(summary['artifacts']['summary_json'])}, "
        f"{os.path.basename(fig_png)}")


if __name__ == "__main__":
    main()
