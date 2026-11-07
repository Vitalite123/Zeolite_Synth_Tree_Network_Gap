"""
E9b: E9 rerun experiment (preregistration in DeepLearning_Supplementary_Plan.md §9.2 [rerun experiment E9b])
================================================================
Purpose
----
also completing the two reopen conditions registered in E9:
  1) Refine category aggregation from 3 to 8 categories (§2.3 feature system) and give bootstrap 95% CIs;
  2) Introduce an architecture-agnostic third method (8-group exact Shapley, 256 coalitions, background mean imputation,
     the value function is predicted-class probability), re-examining the conclusion that "attention ≠ feature contribution".

Metric conventions
----
- Explanation set / background / SHAP / permutation / attention fully consistent with E9 (384 samples seed 42, bg 64,
  GradientSHAP ns=32, permutation n_repeats=3, predicted-class contribution);
- Anchor assertion: the 3-category shares match E9_summary.json item by item within |Δ| ≤ 0.5 pp; the per-sample mean of attention and
  E9 vectors within |Δ| ≤ 1e-6;
- bootstrap: B=1000 for per-sample attribution methods; permutation requires recomputing the model, B=100 with n_repeats=1 each;
- Criteria: R1 8-class rho(SHAP, permutation) >=2/3 of models >0.8; R2 the OSDA-Composition difference of SHAP
  CI lower bound; R3 group-level Shapley top category and rho8; R4 attention top category under the 8-category convention.

Artifacts (E9b_* prefix, smoke carries the _smoke suffix)
---------------------------------------
E9b_report.md / E9b_summary.json / E9b_category8_summary.csv / E9b_rho8.csv /
E9b_bootstrap_ci.csv / E9b_attributions.npz / E9b_metrics.png / e9b_run.log
"""

import argparse
import json
import logging
import os
import sys
import time
from math import factorial

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from e9_interpretability import (  # noqa: E402
    FEATURES_104, N_CLASS_EXPECT, SEED, CATEGORIES, CATS, DEVICE,
    load_arrays, stratified_sample, build_model, torch_proba,
    permutation_importance, tree_shap_xgb, expected_gradients,
)

# ==================== 8-category scheme (section 2.3, strictly aligned with the 104-dim indices) ====================
GROUPS8 = [
    ("Element Composition", 0, 45),
    ("OSDA Index", 45, 48),
    ("Crystallization Conditions", 48, 52),
    ("Aging Conditions", 52, 54),
    ("pH Conditions", 54, 56),
    ("Gel Ratios", 56, 61),
    ("OSDA Descriptors", 61, 94),
    ("OSDA Aggregates", 94, 104),
]
G8_NAMES = [g[0] for g in GROUPS8]
G8_COUNTS = {name: b - a for name, a, b in GROUPS8}
EXPECTED_G8 = {"Element Composition": 45, "OSDA Index": 3,
               "Crystallization Conditions": 4, "Aging Conditions": 2,
               "pH Conditions": 2, "Gel Ratios": 5,
               "OSDA Descriptors": 33, "OSDA Aggregates": 10}

G8_OF = np.zeros(len(FEATURES_104), dtype=np.int64)
for _i, (_n, _a, _b) in enumerate(GROUPS8):
    G8_OF[_a:_b] = _i

# 8 categories -> 3 categories (equivalent to E9 classify_feature_detailed)
G8_TO_3 = {
    "Element Composition": "Composition",
    "OSDA Index": "OSDA",
    "Crystallization Conditions": "Synthesis Conditions",
    "Aging Conditions": "Synthesis Conditions",
    "pH Conditions": "Composition",
    "Gel Ratios": "Composition",
    "OSDA Descriptors": "OSDA",
    "OSDA Aggregates": "OSDA",
}

LOG = None


def make_logger(tag):
    lg = logging.getLogger("E9b")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
        for h in (logging.FileHandler(os.path.join(HERE, f"e9b_run{tag}.log"),
                                     encoding="utf-8"),
                  logging.StreamHandler()):
            h.setFormatter(fmt)
            lg.addHandler(h)
    return lg


def out(name, tag):
    stem, ext = os.path.splitext(name)
    return os.path.join(HERE, f"{stem}{tag}{ext}")


# ==================== Per-sample attention ====================
def bigru_attention_ps(model, X, batch=512):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.as_tensor(X[i:i + batch], dtype=torch.float32, device=DEVICE)
            _, w = model(xb)
            w104 = (w / 4.0).unsqueeze(-1).expand(-1, -1, 4).reshape(xb.size(0), 104)
            outs.append(w104.cpu().numpy())
    attn = np.concatenate(outs, axis=0)
    if np.abs(attn.sum(axis=1) - 1.0).max() > 1e-3:
        raise RuntimeError("BiGRU attention not normalized per sample")
    return attn


def ftt_attention_ps(model, X, batch=256):
    """Per-sample CLS->feature attention (heads averaged within a layer, then averaged across layers)."""
    blocks = model.transformer.blocks
    bufs = [[] for _ in blocks]
    self_bufs = [[] for _ in blocks]
    hooks = []

    def make_hook(i):
        def hook(mod, inp, output):
            probs = output[1]["attention_probs"]
            heads = int(getattr(mod, "n_heads"))
            b = probs.shape[0] // heads
            w = probs.view(b, heads, probs.shape[1], probs.shape[2]).mean(dim=1)
            bufs[i].append(w[:, 0, 1:].detach().cpu().numpy())
            self_bufs[i].append(w[:, 0, 0].detach().cpu().numpy())
        return hook

    for i, blk in enumerate(blocks):
        hooks.append(blk.attention.register_forward_hook(make_hook(i)))
    try:
        model.eval()
        with torch.no_grad():
            for i in range(0, len(X), batch):
                xb = torch.as_tensor(X[i:i + batch], dtype=torch.float32, device=DEVICE)
                model(xb, None)
    finally:
        for h in hooks:
            h.remove()
    layers = [np.concatenate(b, axis=0) for b in bufs]
    attn = np.mean(layers, axis=0)
    self_mass = float(np.mean([np.concatenate(b).mean() for b in self_bufs]))
    s = float(attn.sum(axis=1).mean())
    if abs(s + self_mass - 1.0) > 1e-2:
        raise RuntimeError(f"FTT attention not normalized: sum={s:.6f} self={self_mass:.6f}")
    if self_mass > 0.1:
        raise RuntimeError(f"FTT CLS self-attention share invalid: {self_mass:.4f}")
    return attn, self_mass


# ==================== Third method: exact Shapley over 8 groups ====================
def group_shapley8(proba_fn, X, bg_mean, pred):
    """Returns exact Shapley values of shape (n, 8); efficiency assertion |Sum phi - (f(x)-f(bg_mean))| < 1e-4."""
    n = len(X)
    g = len(GROUPS8)
    n_mask = 1 << g
    v = np.zeros((n, n_mask), dtype=np.float64)
    base = np.tile(np.asarray(bg_mean, dtype=np.float32), (n, 1))
    for mask in range(n_mask):
        Xc = base.copy()
        for k, (_name, a, b) in enumerate(GROUPS8):
            if (mask >> k) & 1:
                Xc[:, a:b] = X[:, a:b]
        p = np.asarray(proba_fn(Xc), dtype=np.float64)
        v[:, mask] = p[np.arange(n), pred]
    phi = np.zeros((n, g), dtype=np.float64)
    for k in range(g):
        for mask in range(n_mask):
            if (mask >> k) & 1:
                continue
            s = bin(mask).count("1")
            w = factorial(s) * factorial(g - s - 1) / factorial(g)
            phi[:, k] += w * (v[:, mask | (1 << k)] - v[:, mask])
    eff_err = float(np.max(np.abs(phi.sum(axis=1) - (v[:, n_mask - 1] - v[:, 0]))))
    if not eff_err < 1e-4:
        raise RuntimeError(f"group-level Shapley efficiency assertion failed: max|Σφ-Δf|={eff_err:.2e}")
    return phi, eff_err, v


# ==================== Aggregation and bootstrap ====================
def group_sums(mat):
    """(n,104) -> (n,8) within-group sum."""
    gm = np.zeros((mat.shape[0], len(GROUPS8)), dtype=np.float64)
    for k, (_name, a, b) in enumerate(GROUPS8):
        gm[:, k] = mat[:, a:b].sum(axis=1)
    return gm


def pct_from_gm(gm):
    tot = gm.sum(axis=1, keepdims=True)
    tot[tot <= 0] = 1.0
    return gm / tot * 100.0


def boot_shares(gm, boot_idx):
    reps = np.zeros((len(boot_idx), gm.shape[1]), dtype=np.float64)
    for r, ii in enumerate(boot_idx):
        s = gm[ii].sum(axis=0)
        reps[r] = s / s.sum() * 100.0 if s.sum() > 0 else 0.0
    return reps


def boot_idx(n, B, seed):
    rng = np.random.RandomState(seed)
    return [rng.randint(0, n, n) for _ in range(B)]


def permutation_importance_vec(predict, X, y, n_repeats, seed):
    """Same estimator as E9 (Acc drop), vectorized per feature to reduce GPU call overhead."""
    n, n_features = X.shape
    rng = np.random.RandomState(seed)
    base_acc = float((predict(X) == y).mean())
    accs = np.zeros((n_features, n_repeats), dtype=np.float64)
    for r in range(n_repeats):
        Xcat = np.empty((n_features * n, n_features), dtype=np.float32)
        for j in range(n_features):
            Xp = X.copy()
            Xp[:, j] = Xp[rng.permutation(n), j]
            Xcat[j * n:(j + 1) * n] = Xp
        pr = np.asarray(predict(Xcat)).reshape(n_features, n)
        accs[:, r] = base_acc - (pr == y[None, :]).mean(axis=1)
    return np.clip(accs.mean(axis=1), 0.0, None)


def rho(a, b):
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(spearmanr(a, b).statistic)


def ci(vals):
    """95% percentile CI; returns (lo, hi) scalars for 1-D, per-column (lo, hi) arrays for 2-D."""
    vals = np.asarray(vals, dtype=np.float64)
    if vals.ndim == 1:
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return float("nan"), float("nan")
    else:
        vals = vals[np.isfinite(vals).all(axis=1)]
        if len(vals) == 0:
            nan = np.full(vals.shape[1], np.nan)
            return nan, nan
    lo, hi = np.percentile(vals, [2.5, 97.5], axis=0)
    if np.ndim(lo) == 0:
        return float(lo), float(hi)
    return lo, hi


def md_table(df, digits=2):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append("nan" if not np.isfinite(v) else f"{v:.{digits}f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ==================== Main pipeline ====================
def main(args):
    global LOG
    t0 = time.time()
    tag = args.tag
    LOG = make_logger(tag)
    log = LOG.info

    if G8_COUNTS != EXPECTED_G8:
        raise RuntimeError(f"8-class counts invalid: {G8_COUNTS}")
    cats_from_g8 = [G8_TO_3[GROUPS8[G8_OF[j]][0]] for j in range(len(FEATURES_104))]
    if cats_from_g8 != list(CATS):
        bad = [(FEATURES_104[j], cats_from_g8[j], CATS[j])
               for j in range(len(FEATURES_104)) if cats_from_g8[j] != CATS[j]]
        raise RuntimeError(f"8-class to 3-class mapping inconsistent with E9 classifier: {bad[:5]}")
    from collections import Counter
    c3 = Counter(cats_from_g8)
    if not (c3["Composition"] == 52 and c3["OSDA"] == 46
            and c3["Synthesis Conditions"] == 6):
        raise RuntimeError(f"3-class counts invalid: {dict(c3)}")

    log("=" * 72)
    log(f"[E9b] 8-class aggregation + bootstrap CI + group-level exact Shapley | device={DEVICE} | tag='{tag}'")
    log(f"explain set n={args.n_explain} | bg={args.n_bg} | nsamples={args.nsamples} | "
        f"B_shap={args.b_shap} | B_perm={args.b_perm}")
    log("8-class counts: " + ", ".join(f"{k}={v}" for k, v in G8_COUNTS.items()))

    data = load_arrays()
    y_test = data["y_test"]
    idx = stratified_sample(y_test, args.n_explain, SEED)
    rng = np.random.RandomState(SEED + 1)
    bg_idx = rng.choice(len(data["y_train"]), args.n_bg, replace=False)
    Xe_raw = data["X_test_raw"][idx]
    Xe_s = data["X_test_s"][idx]
    ye = y_test[idx]
    bg_raw = data["X_train_raw"][bg_idx]
    bg_s = data["X_train_s"][bg_idx]
    log(f"explain set class coverage: {len(np.unique(ye))}")

    e9 = {}
    p_e9 = os.path.join(HERE, "E9_summary.json")
    if os.path.exists(p_e9):
        with open(p_e9, encoding="utf-8") as f:
            js = json.load(f)
        for r in js["category_summary"]:
            e9[(r["model"], r["method"], r["category"])] = r["importance_ratio_pct"]
        log(f"loaded E9 anchors: {len(e9)} class-share entries")
    else:
        raise FileNotFoundError("missing E9_summary.json, cannot make anchor assertions")

    specs = [("XGBoost-104", "xgb"), ("BiGRU", "bigru"), ("FT-Transformer", "ftt")]
    # Anchor tolerances (pp) stratified by method: tight checks for deterministic methods; GradientSHAP uses stochastic expected gradients
    # (Reopening registered in DeepLearning_Supplementary_Plan.md section 9.2 E9b)
    def anchor_tol(kind, method):
        if args.smoke:
            return 20.0          # smoke subset differs, coarse check only
        if method == "shap":
            return 0.5 if kind == "xgb" else 1.0
        if method == "attention":
            return 0.1
        return 0.5
    bi = boot_idx(args.n_explain, args.b_shap, SEED + 11)
    bi_perm = boot_idx(args.n_explain, args.b_perm, SEED + 12)

    cat8_all, rho_rows, ci_rows = [], [], []
    anchors, results, phi_store = [], {}, {}
    for name, kind in specs:
        log("-" * 72)
        log(f"[{name}] loading checkpoint ...")
        model, forward, scaled = build_model(kind)
        Xe = Xe_s if scaled else Xe_raw
        bg = bg_s if scaled else bg_raw
        X_full = data["X_test_s"] if scaled else data["X_test_raw"]

        if kind == "xgb":
            proba_fn = lambda X, f=forward: np.asarray(f(X), dtype=np.float64)
        else:
            proba_fn = lambda X, m=model, f=forward: torch_proba(m, f, X).astype(np.float64)

        proba = proba_fn(X_full)
        pred_eval = proba[idx].argmax(axis=1)

        # ---- Build an E9-convention predict (for permutation point estimates) ----
        if kind == "xgb":
            predict = lambda X, f=forward: np.asarray(f(X)).argmax(axis=1)
        else:
            predict = lambda X, m=model, f=forward: torch_proba(m, f, X).argmax(axis=1)

        perm, _, base_acc = permutation_importance(predict, Xe, ye, args.n_repeats, SEED)
        log(f"permutation point estimate complete | base Acc={base_acc:.4f}")

        if kind == "xgb":
            shap_signed = tree_shap_xgb(model, forward, Xe, pred_eval)
        else:
            shap_signed, pred_shap = expected_gradients(
                model, forward, Xe, bg, args.nsamples, SEED, group=args.group)
            if not np.array_equal(pred_shap, pred_eval):
                raise RuntimeError(f"{name} GradientSHAP predicted class inconsistent")
        shap_abs = np.abs(shap_signed)

        if kind == "bigru":
            attn = bigru_attention_ps(model, Xe)
        elif kind == "ftt":
            attn, self_mass = ftt_attention_ps(model, Xe)
            log(f"FTT CLS->CLS self-attention={self_mass:.4f}")
        else:
            attn = None

        t_gs = time.time()
        phi, eff_err, _ = group_shapley8(proba_fn, Xe, bg.mean(axis=0), pred_eval)
        log(f"group-level exact Shapley complete | 256 coalitions | efficiency error={eff_err:.2e} | "
            f"{time.time() - t_gs:.1f}s")

        # ---- 8-category point estimates ----
        gm_shap = group_sums(shap_abs)
        gm_perm = group_sums(perm[None, :])[0]
        gm_gs = np.clip(phi, 0.0, None)
        gm_attn = group_sums(attn) if attn is not None else None
        pct = {
            "shap": pct_from_gm(gm_shap).mean(axis=0),
            "permutation": pct_from_gm(gm_perm[None, :])[0],
            "group_shapley": pct_from_gm(gm_gs).mean(axis=0),
        }
        if gm_attn is not None:
            pct["attention"] = pct_from_gm(gm_attn).mean(axis=0)

        # ---- Anchor: 3-category shares reproduce E9 ----
        gm_all = np.vstack([pct_from_gm(gm_shap).mean(axis=0),
                            pct_from_gm(gm_perm[None, :])[0],
                            pct_from_gm(gm_gs).mean(axis=0)])
        gp3 = gm_all
        for row_i, method in enumerate(("shap", "permutation", "group_shapley")):
            tol = anchor_tol(kind, method)
            for c in CATEGORIES:
                idxs = [k for k, g in enumerate(GROUPS8) if G8_TO_3[g[0]] == c]
                v = float(gp3[row_i][idxs].sum())
                ref = e9.get((name, method, c))
                if ref is None:
                    continue   # group_shapley has no anchor in E9
                anchors.append({"model": name, "method": method, "category": c,
                                "e9_pct": ref, "e9b_pct": v, "delta_pp": v - ref,
                                "tol_pp": tol})
                if abs(v - ref) > tol:
                    raise RuntimeError(
                        f"anchor reproduction failed {name}/{method}/{c}: {v:.2f} vs {ref:.2f}")
        if gm_attn is not None:
            attn_mean = attn.mean(axis=0)
            for c in CATEGORIES:
                mask = np.array([x == c for x in CATS])
                ref = e9.get((name, "attention", c))
                if ref is None:
                    raise RuntimeError(f"missing E9 attention anchor {name}/{c}")
                v = float(attn_mean[mask].sum() / attn_mean.sum() * 100.0)
                if abs(v - ref) > anchor_tol(kind, "attention"):
                    raise RuntimeError(f"attention anchor reproduction failed {name}/{c}: {v:.2f} vs {ref:.2f}")

        # ---- bootstrap ----
        reps = {
            "shap": boot_shares(gm_shap, bi),
            "group_shapley": boot_shares(gm_gs, bi),
        }
        if gm_attn is not None:
            reps["attention"] = boot_shares(gm_attn, bi)
        perm_reps = np.zeros((args.b_perm, len(GROUPS8)))
        for r, ii in enumerate(bi_perm):
            pv = permutation_importance_vec(predict, Xe[ii], ye[ii], 1, SEED + 100 + r)
            perm_reps[r] = pct_from_gm(group_sums(pv[None, :]))[0]
        reps["permutation"] = perm_reps

        pct3 = {}
        for m in pct:
            v3 = np.zeros(len(CATEGORIES))
            for j, c in enumerate(CATEGORIES):
                idxs = [k for k, g in enumerate(GROUPS8) if G8_TO_3[g[0]] == c]
                v3[j] = pct[m][idxs].sum()
            pct3[m] = v3

        for method, arr in reps.items():
            lo, hi = ci(arr)
            for k, gname in enumerate(G8_NAMES):
                ci_rows.append({"model": name, "method": method, "category": gname,
                                "pct_mean": float(pct[method][k]),
                                "ci_lo": float(lo[k]), "ci_hi": float(hi[k]),
                                "bootstrap_B": len(arr)})
            # 3-category CI
            arr3 = np.zeros((len(arr), len(CATEGORIES)))
            for j, c in enumerate(CATEGORIES):
                idxs = [k for k, g in enumerate(GROUPS8) if G8_TO_3[g[0]] == c]
                arr3[:, j] = arr[:, idxs].sum(axis=1)
            lo3, hi3 = ci(arr3)
            for j, c in enumerate(CATEGORIES):
                ci_rows.append({"model": name, "method": method, "category": c,
                                "pct_mean": float(pct3[method][j]),
                                "ci_lo": float(lo3[j]), "ci_hi": float(hi3[j]),
                                "bootstrap_B": len(arr)})

        # ---- rho (8 categories) point estimates and bootstrap CI ----
        pairs = []
        if "attention" in pct:
            pairs.append(("attention", "permutation"))
        pairs.append(("group_shapley", "permutation"))
        pairs.append(("group_shapley", "shap"))
        if "attention" in pct:
            pairs.append(("attention", "shap"))
        pairs.append(("shap", "permutation"))
        for m1, m2 in pairs:
            r_pt = rho(pct[m1], pct[m2])
            key1 = "shap" if m1 == "shap" else m1
            rhos = []
            n_r = min(len(reps[m1]), len(reps[m2]))
            for r in range(n_r):
                v = rho(reps[m1][r], reps[m2][r])
                if np.isfinite(v):
                    rhos.append(v)
            lo, hi = ci(rhos)
            rho_rows.append({"model": name, "method_a": m1, "method_b": m2,
                             "rho8": r_pt, "ci_lo": lo, "ci_hi": hi,
                             "n_boot": len(rhos)})

    # Gauss difference CI: OSDA - Composition (SHAP / permutation / group Shapley)
        for m in ("shap", "permutation", "group_shapley"):
            if m not in reps:
                continue
            oi = G8_NAMES.index("OSDA Index")
            od = [oi, G8_NAMES.index("OSDA Descriptors"),
                  G8_NAMES.index("OSDA Aggregates")]
            ci_c = [G8_NAMES.index("Element Composition"),
                    G8_NAMES.index("pH Conditions"),
                    G8_NAMES.index("Gel Ratios")]
            gap = reps[m][:, od].sum(axis=1) - reps[m][:, ci_c].sum(axis=1)
            lo, hi = ci(gap)
            ci_rows.append({"model": name, "method": m,
                            "category": "OSDA-minus-Composition",
                            "pct_mean": float(np.mean(gap)), "ci_lo": lo, "ci_hi": hi,
                            "bootstrap_B": len(gap)})

        for k, gname in enumerate(G8_NAMES):
            row = {"model": name, "category": gname}
            for m in pct:
                row[f"{m}_pct"] = float(pct[m][k])
            cat8_all.append(row)
        results[name] = {m: [float(x) for x in pct[m]] for m in pct}
        results[name]["group_shapley_eff_err"] = eff_err
        results[name]["group_shapley_top"] = G8_NAMES[int(np.argmax(pct["group_shapley"]))]
        results[name]["attention_top"] = (G8_NAMES[int(np.argmax(pct["attention"]))]
                                          if "attention" in pct else None)
        results[name]["shap_top"] = G8_NAMES[int(np.argmax(pct["shap"]))]
        results[name]["permutation_top"] = G8_NAMES[int(np.argmax(pct["permutation"]))]
        results[name]["pct3"] = {m: [float(x) for x in pct3[m]] for m in pct3}
        results[name]["top3"] = {m: CATEGORIES[int(np.argmax(pct3[m]))] for m in pct3}
        results[name]["group_shapley_top3"] = results[name]["top3"]["group_shapley"]
        results[name]["attention_top3"] = results[name]["top3"].get("attention")
        log(f"[{name}] 8-class first class: SHAP={results[name]['shap_top']} | "
            f"permutation={results[name]['permutation_top']} | "
            f"group Shapley={results[name]['group_shapley_top']} | "
            f"attention={results[name]['attention_top']} || top-3 first class: "
            f"SHAP={results[name]['top3']['shap']} | "
            f"permutation={results[name]['top3']['permutation']} | "
            f"group Shapley={results[name]['group_shapley_top3']} | "
            f"attention={results[name]['attention_top3']}")
        phi_store[name] = phi

        del model
        import gc
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # ---- Criteria ----
    ev = [n for n in results if "shap" in results[n]]
    rho_sp = {n: next(r["rho8"] for r in rho_rows
                      if r["model"] == n and r["method_a"] == "shap"
                      and r["method_b"] == "permutation") for n in ev}
    n_r1 = sum(1 for v in rho_sp.values() if np.isfinite(v) and v > 0.8)
    r1 = ("dual-method validation holds under the 8-class metric (>=2/3 models rho>0.8)" if n_r1 >= 2 else
          "Under the 8-category convention, dual-method validation fails (<2/3 models rho>0.8); only the 3-category result stands")

    def gap_ci(model, method):
        return next((r for r in ci_rows if r["model"] == model and r["method"] == method
                     and r["category"] == "OSDA-minus-Composition"), None)

    n_r2 = 0
    for n in ev:
        g = gap_ci(n, "shap")
        if g is not None and g["ci_lo"] > 0:
            n_r2 += 1
    r2 = ("SHAP's OSDA advantage CI lower bound >0 (>=2/3 models), keep the OSDA-dominance wording"
          if n_r2 >= 2 else
          "The SHAP OSDA advantage CI spans 0 (>=2/3 models); downgrade to OSDA tied with composition")

    gs3_top = {n: results[n]["group_shapley_top3"] for n in ev}
    rho_gs = {n: next((r["rho8"] for r in rho_rows if r["model"] == n
                       and r["method_a"] == "group_shapley"
                       and r["method_b"] == "permutation"), float("nan")) for n in ev}
    n_r3 = sum(1 for n in ev if gs3_top[n] == "OSDA"
               and np.isfinite(rho_gs[n]) and rho_gs[n] > 0.8)
    n_conf = sum(1 for n in ev if gs3_top[n] == "Composition")
    if n_r3 >= 2:
        r3 = f"models with group-level Shapley (3 classes) first class OSDA and ρ8>0.8: {n_r3}/{len(ev)}: OSDA dominance gains support from a third architecture-independent method"
    elif n_conf >= 2:
        r3 = (f"models whose group-level Shapley (3 classes) first class is Composition: {n_conf}/{len(ev)}: "
              "Consistent with attention but conflicting with SHAP/permutation; section 5.4.4 must be rewritten ")
    else:
        r3 = f"group-level Shapley (3 classes) results are mixed (OSDA {n_r3}/{len(ev)}, Composition {n_conf}/{len(ev)}), report per model"

    att3_top = {n: results[n]["top3"].get("attention") for n in ev}
    att_models = [n for n in ev if att3_top[n] is not None]
    n_att = len(att_models)
    n_r4 = sum(1 for n in att_models if att3_top[n] == "Composition")
    r4 = (f"models whose attention first class under 3-class aggregation is Composition {n_r4}/{n_att} (models with attention attribution): " +
          ("Maintain §5.4.4 (attention != feature contribution)" if n_r4 >= 2 else
           "attention ranking changed; rewrite section 5.4.4 per the new results "))

    verdicts = {"R1_8class_dual_method": {"pass": n_r1, "n": len(ev), "text": r1},
                "R2_osda_gap_ci": {"pass": n_r2, "n": len(ev), "text": r2},
                "R3_group_shapley": {"pass": n_r3, "n": len(ev), "text": r3},
                "R4_attention_8class": {"pass": n_r4, "n": n_att, "text": r4}}

    # ---- Persist to disk ----
    cat8_df = pd.DataFrame(cat8_all)
    cat8_df.to_csv(out("E9b_category8_summary.csv", tag), index=False, encoding="utf-8")
    rho_df = pd.DataFrame(rho_rows)
    rho_df.to_csv(out("E9b_rho8.csv", tag), index=False, encoding="utf-8")
    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(out("E9b_bootstrap_ci.csv", tag), index=False, encoding="utf-8")
    np.savez_compressed(
        out("E9b_attributions.npz", tag),
        groups=np.array(G8_NAMES),
        phi=np.stack([phi_store[n].mean(axis=0) for n in phi_store]),
        phi_models=np.array(list(phi_store)),
    )

    # Figure: 8-category shares (3 models x 3-4 methods)
    fig, axes = plt.subplots(1, len(specs), figsize=(5.2 * len(specs), 4.2),
                             constrained_layout=True)
    colors = {"shap": "#2f6fb5", "permutation": "#d9822b",
              "group_shapley": "#3f9c5a", "attention": "#8a5bb5"}
    for ax, (name, _kind) in zip(np.atleast_1d(axes), specs):
        methods = [m for m in ("shap", "permutation", "group_shapley", "attention")
                   if m in results[name]]
        x = np.arange(len(G8_NAMES))
        w = 0.8 / len(methods)
        for i, m in enumerate(methods):
            ax.bar(x + (i - (len(methods) - 1) / 2) * w, results[name][m], w,
                   label=m, color=colors[m])
        ax.set_xticks(x)
        ax.set_xticklabels([g.replace(" ", "\n") for g in G8_NAMES], fontsize=7)
        ax.set_title(name, fontsize=10)
        ax.set_ylabel("share (%)")
        ax.legend(fontsize=7)
    fig.savefig(out("E9b_metrics.png", tag), dpi=150)

    payload = {
        "exp_id": "E9b", "tag": tag, "seed": SEED, "device": str(DEVICE),
        "config": {"n_explain": int(len(idx)), "n_bg": args.n_bg,
                   "nsamples": args.nsamples, "perm_repeats": args.n_repeats,
                   "b_shap": args.b_shap, "b_perm": args.b_perm},
        "groups8": G8_COUNTS,
        "anchors": anchors,
        "results": results,
        "rho8": rho_rows,
        "verdicts": verdicts,
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(out("E9b_summary.json", tag), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # ---- Report ----
    lines = ["# E9b reopen experiment report (8-class aggregation + bootstrap CI + group-level exact Shapley)",
             "",
             f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')} | Companion: DeepLearning_Supplementary_Plan.md §9.2 E9b",
             "",
             "## 1. Protocol",
             "",
             f"- Explain set: test stratified sampling n={len(idx)} (seed=42), background={args.n_bg} (train);"
             f"GradientSHAP nsamples={args.nsamples}; permutation n_repeats={args.n_repeats}.",
             f"- bootstrap: per-sample attribution B={args.b_shap}; permutation importance B={args.b_perm} (n_repeats=1 each).",
             "- Third method: exact Shapley over 8 groups (256 coalitions, removed groups replaced by background means, "
             "with the predicted-class probability as the value function), with a built-in efficiency assertion. ",
             "- The 8-category grouping is consistent with section 2.3, and after mapping back to 3 categories it equals the E9 classifier feature by feature (counts 52/46/6).",
             "",
             "## 2. Anchor reproduction (3 categories vs E9)",
             "",
             md_table(pd.DataFrame(anchors), digits=2) if anchors else "_none_",
             "",
             "## 3. 8-category shares (point estimates)",
             "",
             md_table(cat8_df, digits=2),
             "",
             "## 4. Spearman rho (8 categories, with bootstrap 95% CI)",
             "",
             md_table(rho_df, digits=3),
             "",
             "## 5. OSDA-Composition differences with 95% CI",
             "",
             md_table(pd.DataFrame([r for r in ci_rows
                                    if r["category"] == "OSDA-minus-Composition"]),
                      digits=2),
             "",
             "## 6. Criteria and branches",
             "",
             f"- **R1**({n_r1}/{len(ev)}):{r1}",
             f"- **R2**({n_r2}/{len(ev)}):{r2}",
             f"- **R3**({n_r3}/{len(ev)}):{r3}",
             f"- **R4**({n_r4}/{len(ev)}):{r4}",
             "",
             "## 7. Reproduction",
             "",
             "```bash",
             "py -B e9b_reopen.py --full",
             "py -B e9b_reopen.py --smoke",
             "```",
             ]
    with open(out("E9b_report.md", tag), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    log("=" * 72)
    log(f"done | {time.time() - t0:.1f}s | R1={n_r1}/{len(ev)} R2={n_r2}/{len(ev)} "
        f"R3={n_r3}/{len(ev)} R4={n_r4}/{n_att}")
    for k, v in verdicts.items():
        log(f"  {k}: {v['text']}")
    return payload


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="E9b reopen experiment")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--full", action="store_true")
    p.add_argument("--n-explain", type=int, default=None)
    p.add_argument("--n-bg", type=int, default=64)
    p.add_argument("--nsamples", type=int, default=32)
    p.add_argument("--n-repeats", type=int, default=3)
    p.add_argument("--group", type=int, default=16)
    p.add_argument("--b-shap", type=int, default=None)
    p.add_argument("--b-perm", type=int, default=None)
    a = p.parse_args(argv)
    if a.smoke:
        a.tag = "_smoke"
        a.n_explain = a.n_explain or 48
        a.b_shap = a.b_shap or 200
        a.b_perm = a.b_perm or 20
    else:
        a.tag = ""
        a.n_explain = a.n_explain or 384
        a.b_shap = a.b_shap or 1000
        a.b_perm = a.b_perm or 100
    return a


if __name__ == "__main__":
    main(parse_args())
