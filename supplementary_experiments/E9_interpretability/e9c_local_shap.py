"""
E9c: local (sample-level) SHAP and misclassified-sample attribution
======================================
Pre-registration in DeepLearning_Supplementary_Plan.md §9.2 [Reopen experiment E9c] (2026-10-25).

Purpose
----
1) Test whether E9b's 3-category conclusions are a per-sample majority phenomenon (rather than means driven by a few samples);
2) Attribution structure of misclassified samples (pred != true): does the dominant category shift to elemental composition, do attributions become more dispersed;
3) Produce the per-sample evidence table (E9c_local_shap.csv) for §6.2.6 error analysis.

Metric conventions (frozen, consistent with E9/E9b)
--------------------------
- Explanation set n=384 (smoke 48, seed 42 stratified), background 64 (train, seed 43),
  GradientSHAP ns=32, seed 42, model checkpoints same as E9;
- Local attribution: per-sample SHAP for the predicted class; local importance = share of |SHAP| aggregated over 3 / 8 categories;
  dominant class = largest-share class; margin = largest − second; entropy = normalized Shannon entropy;
- bootstrap: sample-level resampling B=1000 (the two groups are resampled separately for between-group differences);
- Anchor: the 3-category average shares of |SHAP| must reproduce E9b pct3.shap (TreeSHAP 0.5 pp /
  GradientSHAP 1.0 pp;smoke 20 pp).

Criteria (pre-registered)
--------------
L1 models with per-sample OSDA dominance share > 50% >= 2/3;
L2 models with ΔOSDA = mean(OSDA|mis) − mean(|correct) < 0 and CI upper bound < 0 >= 2/3;
L3 models with Δmargin < 0 and CI upper bound < 0 >= 2/3;
L4 models whose most frequent misclassified samples' dominant 8-cat group is Element Composition >= 2/3.

Artifacts (smoke carries the _smoke suffix)
---------------------------
E9c_report.md / E9c_summary.json / E9c_local_shap.csv / E9c_group_summary.csv /
E9c_misclassified_top.csv / E9c_metrics.png / e9c_run.log
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from e9_interpretability import (  # noqa: E402
    FEATURES_104, SEED, CATEGORIES, CATS, DEVICE,
    load_arrays, stratified_sample, build_model, torch_proba,
    tree_shap_xgb, expected_gradients,
)
from e9b_reopen import (  # noqa: E402
    GROUPS8, G8_NAMES, G8_TO_3, group_sums, pct_from_gm, ci,
)

LOG = None


def make_logger(tag):
    lg = logging.getLogger("E9c")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
        for h in (logging.FileHandler(os.path.join(HERE, f"e9c_run{tag}.log"),
                                     encoding="utf-8"),
                  logging.StreamHandler()):
            h.setFormatter(fmt)
            lg.addHandler(h)
    return lg


def out(name, tag):
    stem, ext = os.path.splitext(name)
    return os.path.join(HERE, f"{stem}{tag}{ext}")


# ==================== Local statistics ====================
def pct3_local(shap_abs):
    """(n,104) |SHAP| -> (n,3) within-sample share of 3 categories (%)."""
    gm = group_sums(shap_abs)
    g3 = np.stack([gm[:, [k for k, g in enumerate(GROUPS8)
                          if G8_TO_3[g[0]] == c]].sum(axis=1)
                   for c in CATEGORIES], axis=1)
    return pct_from_gm(g3)


def entropy_norm(p):
    p = np.clip(p, 1e-12, None)
    p = p / p.sum(axis=1, keepdims=True)
    return -np.sum(p * np.log(p), axis=1) / np.log(p.shape[1])


def boot_mean_ci(v, B, seed):
    v = np.asarray(v, dtype=np.float64)
    if v.size == 0:
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    m = np.array([v[rng.randint(0, len(v), len(v))].mean() for _ in range(B)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_diff_ci(a, b, B, seed):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    d = np.empty(B)
    for r in range(B):
        d[r] = (a[rng.randint(0, len(a), len(a))].mean()
                - b[rng.randint(0, len(b), len(b))].mean())
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def modal(counter):
    if not counter:
        return None
    return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


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


def main(args):
    global LOG
    tag = args.tag
    LOG = make_logger(tag)
    t0 = time.time()
    log = LOG.info
    log("=" * 72)
    log(f"[E9c] local SHAP and misclassified sample attribution | device={DEVICE} | tag='{tag}'")
    log(f"explain set n={args.n_explain} | bg={args.n_bg} | nsamples={args.nsamples} | "
        f"B={args.b_boot} | group={args.group}")

    data = load_arrays()
    y_test = data["y_test"]
    idx = stratified_sample(y_test, args.n_explain, SEED)
    rng = np.random.RandomState(SEED + 1)
    bg_idx = rng.choice(len(data["y_train"]), args.n_bg, replace=False)
    Xe_raw, Xe_s = data["X_test_raw"][idx], data["X_test_s"][idx]
    ye = y_test[idx]
    bg_raw, bg_s = data["X_train_raw"][bg_idx], data["X_train_s"][bg_idx]

    with open(os.path.join(HERE, "E9b_summary.json"), encoding="utf-8") as f:
        e9b = json.load(f)
    log(f"loaded E9b anchors: {len(e9b['results'])} models")

    specs = [("XGBoost-104", "xgb"), ("BiGRU", "bigru"), ("FT-Transformer", "ftt")]
    local_rows, group_rows, top_rows, anchors = [], [], [], []
    models_out, verdict_input = {}, {}

    for name, kind in specs:
        log("-" * 72)
        log(f"[{name}] loading checkpoint ...")
        model, forward, scaled = build_model(kind)
        Xe = Xe_s if scaled else Xe_raw
        bg = bg_s if scaled else bg_raw

        proba = (np.asarray(forward(Xe), dtype=np.float64) if kind == "xgb"
                 else torch_proba(model, forward, Xe).astype(np.float64))
        pred = proba.argmax(axis=1)

        if kind == "xgb":
            shap_signed = tree_shap_xgb(model, forward, Xe, pred)
        else:
            shap_signed, pred_shap = expected_gradients(
                model, forward, Xe, bg, args.nsamples, SEED, group=args.group)
            if not np.array_equal(pred_shap, pred):
                raise RuntimeError(f"{name} GradientSHAP predicted class inconsistent")
        shap_abs = np.abs(shap_signed)
        log("SHAP complete")

        # ---- Anchor: 3-category mean shares reproduce E9b ----
        ref3 = np.asarray(e9b["results"][name]["pct3"]["shap"], dtype=np.float64)
        p3 = pct3_local(shap_abs)
        mean3 = p3.mean(axis=0)
        tol = 20.0 if args.smoke else (0.5 if kind == "xgb" else 1.0)
        for j, c in enumerate(CATEGORIES):
            d = float(mean3[j] - ref3[j])
            anchors.append({"model": name, "category": c, "e9b_pct": float(ref3[j]),
                            "e9c_pct": float(mean3[j]), "delta_pp": d, "tol_pp": tol})
            if abs(d) > tol:
                raise RuntimeError(f"anchor reproduction failed {name}/{c}: {mean3[j]:.2f} vs {ref3[j]:.2f}")
        log(f"anchor passed | 3-class means={np.round(mean3, 2).tolist()} | tolerance={tol} pp")

        # ---- Local quantities ----
        gm8 = group_sums(shap_abs)
        p8 = pct_from_gm(gm8)
        top3 = np.array([CATEGORIES[i] for i in p3.argmax(axis=1)])
        top8 = np.array([G8_NAMES[i] for i in p8.argmax(axis=1)])
        sort3 = np.sort(p3, axis=1)
        margin = sort3[:, -1] - sort3[:, -2]
        ent3 = entropy_norm(p3)
        correct = pred == ye
        signed3 = np.stack([shap_signed[:, [k for k, g in enumerate(GROUPS8)
                                            if G8_TO_3[g[0]] == c]].sum(axis=1)
                            for c in CATEGORIES], axis=1)

        total_abs = shap_abs.sum(axis=1)
        for i in range(len(idx)):
            order = np.argsort(-shap_abs[i])[:10]
            feats = "; ".join(f"{FEATURES_104[j]}:{shap_signed[i, j]:+.4f}"
                              for j in order)
            row = {"model": name, "test_index": int(idx[i]), "pred": int(pred[i]),
                   "true": int(ye[i]), "correct": bool(correct[i]),
                   "share_composition": float(p3[i, 0]),
                   "share_osda": float(p3[i, 1]),
                   "share_synthesis": float(p3[i, 2]),
                   "margin3": float(margin[i]), "entropy3": float(ent3[i]),
                   "top3": top3[i], "top8": top8[i],
                   "total_abs_shap": float(total_abs[i]),
                   "top10_features": feats}
            for k, g in enumerate(G8_NAMES):
                row[f"share8_{g}"] = float(p8[i, k])
            local_rows.append(row)

        for grp, mask in (("correct", correct), ("mis", ~correct)):
            n_g = int(mask.sum())
            if n_g == 0:
                continue
            group_rows.append({
                "model": name, "group": grp, "n": n_g,
                "composition": float(p3[mask, 0].mean()),
                "osda": float(p3[mask, 1].mean()),
                "synthesis": float(p3[mask, 2].mean()),
                "margin3": float(margin[mask].mean()),
                "entropy3": float(ent3[mask].mean()),
                "signed_composition": float(signed3[mask, 0].mean()),
                "signed_osda": float(signed3[mask, 1].mean()),
                "signed_synthesis": float(signed3[mask, 2].mean()),
            })

        mis = ~correct
        n_mis = int(mis.sum())
        acc = float(correct.mean())
        osda_top_share = float((top3 == "OSDA").mean())
        comp_top_share = float((top3 == "Composition").mean())
        l1_lo, l1_hi = boot_mean_ci((top3 == "OSDA").astype(float), args.b_boot, SEED + 21)
        d_osda = (float(p3[mis, 1].mean() - p3[correct, 1].mean())
                  if n_mis and correct.any() else float("nan"))
        d_margin = (float(margin[mis].mean() - margin[correct].mean())
                    if n_mis and correct.any() else float("nan"))
        d_osda_ci = boot_diff_ci(p3[mis, 1], p3[correct, 1], args.b_boot, SEED + 22)
        d_margin_ci = boot_diff_ci(margin[mis], margin[correct], args.b_boot, SEED + 23)
        c8_mis = Counter(top8[mis])
        modal8 = modal(c8_mis)
        c8_cor = Counter(top8[correct])
        c3_mis = Counter(top3[mis])
        c3_cor = Counter(top3[correct])

        for i in np.where(mis)[0][:10]:
            order = np.argsort(-shap_abs[i])[:5]
            top_rows.append({
                "model": name, "test_index": int(idx[i]), "pred": int(pred[i]),
                "true": int(ye[i]), "top3": top3[i], "top8": top8[i],
                "total_abs_shap": float(total_abs[i]),
                "margin3": float(margin[i]),
                "top5_features": "; ".join(
                    f"{FEATURES_104[j]}:{shap_signed[i, j]:+.4f}" for j in order),
            })

        models_out[name] = {
            "acc_explain": acc, "n_correct": int(correct.sum()), "n_mis": n_mis,
            "mean3_shap": [float(x) for x in mean3],
            "osda_top_share": osda_top_share,
            "osda_top_ci": [l1_lo, l1_hi],
            "composition_top_share": comp_top_share,
            "delta_osda_mis_minus_correct": d_osda,
            "delta_osda_ci": [d_osda_ci[0], d_osda_ci[1]],
            "delta_margin_mis_minus_correct": d_margin,
            "delta_margin_ci": [d_margin_ci[0], d_margin_ci[1]],
            "top3_counts_mis": dict(c3_mis),
            "top3_counts_correct": dict(c3_cor),
            "top8_counts_mis": dict(c8_mis),
            "modal_top8_mis": modal8,
            "top8_counts_correct": dict(c8_cor),
            "modal_top8_correct": modal(c8_cor),
            "anchor_tol_pp": tol,
        }
        verdict_input[name] = models_out[name]
        log(f"[{name}] Acc={acc:.4f} | misclassified={n_mis}/{len(idx)} | OSDA dominance share="
            f"{osda_top_share:.3f} ({l1_lo:.3f}-{l1_hi:.3f}) | ΔOSDA={d_osda:+.2f} pp "
            f"[{d_osda_ci[0]:+.2f},{d_osda_ci[1]:+.2f}] | Δmargin={d_margin:+.2f} "
            f"[{d_margin_ci[0]:+.2f},{d_margin_ci[1]:+.2f}] | modal-8 misclassified classes={modal8}")

        del model
        import gc
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # ---- Criteria ----
    ev = list(verdict_input)
    n_l1 = sum(1 for n in ev if verdict_input[n]["osda_top_share"] > 0.5)
    n_l2 = sum(1 for n in ev
               if np.isfinite(verdict_input[n]["delta_osda_mis_minus_correct"])
               and verdict_input[n]["delta_osda_mis_minus_correct"] < 0
               and verdict_input[n]["delta_osda_ci"][1] < 0)
    n_l3 = sum(1 for n in ev
               if np.isfinite(verdict_input[n]["delta_margin_mis_minus_correct"])
               and verdict_input[n]["delta_margin_mis_minus_correct"] < 0
               and verdict_input[n]["delta_margin_ci"][1] < 0)
    n_l4 = sum(1 for n in ev if verdict_input[n]["modal_top8_mis"] == "Element Composition")

    l1 = (f"models with OSDA local-dominance share >50%: {n_l1}/{len(ev)}: " +
          ("OSDA dominance is a per-sample majority phenomenon" if n_l1 >= 2 else
           "OSDA dominance is a mean-level effect, not per-sample majority (must be noted in section 5.3.7) "))
    l2 = (f"models with ΔOSDA<0 and CI upper bound<0: {n_l2}/{len(ev)}: " +
          ("Misclassifications correlate weakly with OSDA evidence" if n_l2 >= 2 else
           "misclassification is not attributable to weak OSDA evidence (rephrase as over-attribution/misleading hypothesis and keep both explanations) "))
    l3 = (f"models with Δmargin<0 and CI upper bound<0: {n_l3}/{len(ev)}: " +
          ("Misclassifications come with more dispersed attributions (margin can serve as an uncertainty proxy)" if n_l3 >= 2 else
           "Margin is not supported as an uncertainty proxy; drop that explanation "))
    l4 = (f"models whose dominant group among modal-8 misclassified classes is Element Composition: {n_l4}/{len(ev)}: " +
          ("Misclassifications are dominated by elemental-composition evidence, corroborating the E9b 8-category conclusion" if n_l4 >= 2 else
           "the misclassification dominant group is not Element Composition; record the actual distribution without asserting an association "))

    verdicts = {"L1_local_osda_majority": {"pass": n_l1, "n": len(ev), "text": l1},
                "L2_mis_osda_weak": {"pass": n_l2, "n": len(ev), "text": l2},
                "L3_mis_diffuse": {"pass": n_l3, "n": len(ev), "text": l3},
                "L4_mis_top8": {"pass": n_l4, "n": len(ev), "text": l4}}

    # ---- Persist to disk ----
    df_local = pd.DataFrame(local_rows)
    df_local.to_csv(out("E9c_local_shap.csv", tag), index=False, encoding="utf-8-sig")
    pd.DataFrame(group_rows).to_csv(out("E9c_group_summary.csv", tag), index=False,
                                   encoding="utf-8-sig")
    pd.DataFrame(top_rows).to_csv(out("E9c_misclassified_top.csv", tag), index=False,
                                 encoding="utf-8-sig")
    pd.DataFrame(anchors).to_csv(out("E9c_anchors.csv", tag), index=False,
                                encoding="utf-8-sig")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    x = np.arange(len(ev))
    vals = [verdict_input[n]["osda_top_share"] for n in ev]
    lo = [verdict_input[n]["osda_top_share"] - verdict_input[n]["osda_top_ci"][0] for n in ev]
    hi = [verdict_input[n]["osda_top_ci"][1] - verdict_input[n]["osda_top_share"] for n in ev]
    ax.bar(x, vals, color="#4C72B0", width=0.55)
    ax.errorbar(x, vals, yerr=[lo, hi], fmt="none", ecolor="k", capsize=4)
    ax.axhline(0.5, ls="--", c="crimson", lw=1)
    ax.set_xticks(x, ev, fontsize=8)
    ax.set_ylabel("OSDA locally dominant sample share")
    ax.set_title("L1 per-sample majority (dashed=0.5)")

    ax = axes[0, 1]
    w = 0.35
    d1 = [verdict_input[n]["delta_osda_mis_minus_correct"] for n in ev]
    d2 = [verdict_input[n]["delta_margin_mis_minus_correct"] for n in ev]
    c1 = [[d1[i] - verdict_input[ev[i]]["delta_osda_ci"][0] for i in range(len(ev))],
          [verdict_input[ev[i]]["delta_osda_ci"][1] - d1[i] for i in range(len(ev))]]
    c2 = [[d2[i] - verdict_input[ev[i]]["delta_margin_ci"][0] for i in range(len(ev))],
          [verdict_input[ev[i]]["delta_margin_ci"][1] - d2[i] for i in range(len(ev))]]
    ax.bar(x - w / 2, d1, w, color="#C44E52", label="ΔOSDA", yerr=c1, capsize=3)
    ax.bar(x + w / 2, d2, w, color="#8172B2", label="Δmargin", yerr=c2, capsize=3)
    ax.axhline(0, c="k", lw=1)
    ax.set_xticks(x, ev, fontsize=8)
    ax.set_ylabel("Misclassified - correct (pp)")
    ax.set_title("L2/L3 between-group difference 95% CI")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    labels = []
    counts = []
    for n in ev:
        c = verdict_input[n]["top8_counts_mis"]
        for g, v in c.items():
            labels.append(f"{n}\n{g}")
            counts.append(v)
    colors = ["#C44E52" if "Element Composition" in lb else
              ("#4C72B0" if "OSDA" in lb else "#999999") for lb in labels]
    ax.barh(np.arange(len(labels)), counts, color=colors)
    ax.set_yticks(np.arange(len(labels)), labels, fontsize=6)
    ax.set_xlabel("Misclassified sample count")
    ax.set_title("L4 dominant 8-cat group distribution of misclassified samples")

    ax = axes[1, 1]
    rng_pos = np.random.RandomState(SEED + 31)
    data_plot, pos, ticklabels = [], [], []
    for i, n in enumerate(ev):
        sub = df_local[df_local["model"] == n]
        for j, (grp, lab) in enumerate((("correct", "correct"), ("mis", "misclassified"))):
            v = sub.loc[sub["correct"] == (grp == "correct"), "share_osda"].values
            if len(v) == 0:
                continue
            data_plot.append(v)
            pos.append(i * 3 + j)
            ticklabels.append(f"{n}\n{lab}")
    bp = ax.boxplot(data_plot, positions=pos, widths=0.8, patch_artist=True,
                    showfliers=False)
    for patch, p in zip(bp["boxes"], pos):
        patch.set_facecolor("#4C72B0" if p % 3 == 0 else "#C44E52")
    ax.set_xticks(pos, ticklabels, fontsize=6)
    ax.set_ylabel("Per-sample OSDA share (%)")
    ax.set_title("OSDA local share: correct vs misclassified")
    fig.tight_layout()
    fig.savefig(out("E9c_metrics.png", tag), dpi=180)
    plt.close(fig)

    payload = {
        "config": {"n_explain": args.n_explain, "n_bg": args.n_bg,
                   "nsamples": args.nsamples, "group": args.group,
                   "b_boot": args.b_boot, "seed": SEED, "device": str(DEVICE),
                   "tag": tag},
        "anchors": anchors, "models": models_out, "verdicts": verdicts,
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(out("E9c_summary.json", tag), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # ---- Report ----
    lines = [
        "# E9c local SHAP and misclassified-sample attribution report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')} | Companion: "
        "DeepLearning_Supplementary_Plan.md section 9.2 E9c",
        "",
        "## 1. Protocol",
        "",
        f"- Explain set n={args.n_explain} (seed=42 stratified); background=64 (train);"
        f"GradientSHAP ns={args.nsamples};bootstrap B={args.b_boot}.",
        "- Local importance = per-sample |SHAP| for the predicted class aggregated into 3-category / 8-category within-sample shares; "
        "Dominant category = the category with the largest margin; margin = largest - second largest; entropy = normalized Shannon. ",
        "- Anchors: the 3-category mean shares reproduce E9b (TreeSHAP 0.5 pp / GradientSHAP 1.0 pp).",
        "",
        "## 2. Anchor reproduction (3-category mean shares vs E9b)",
        "",
        md_table(pd.DataFrame(anchors)),
        "",
        "## 3. Per-sample dominant category and L1",
        "",
        md_table(pd.DataFrame([{
            "model": n, "acc": round(m["acc_explain"], 4),
            "n_correct": m["n_correct"], "n_mis": m["n_mis"],
            "osda_top_share": round(m["osda_top_share"], 4),
            "ci_lo": round(m["osda_top_ci"][0], 4),
            "ci_hi": round(m["osda_top_ci"][1], 4),
            "composition_top_share": round(m["composition_top_share"], 4),
        } for n, m in models_out.items()])),
        "",
        "## 4. Correct vs misclassified group comparison (L2 / L3)",
        "",
        md_table(pd.DataFrame(group_rows), digits=3),
        "",
        md_table(pd.DataFrame([{
            "model": n, "delta_osda": round(m["delta_osda_mis_minus_correct"], 3),
            "osda_ci_lo": round(m["delta_osda_ci"][0], 3),
            "osda_ci_hi": round(m["delta_osda_ci"][1], 3),
            "delta_margin": round(m["delta_margin_mis_minus_correct"], 3),
            "margin_ci_lo": round(m["delta_margin_ci"][0], 3),
            "margin_ci_hi": round(m["delta_margin_ci"][1], 3),
        } for n, m in models_out.items()])),
        "",
        "## 5. 8-category dominant groups of misclassified samples (L4)",
        "",
        md_table(pd.DataFrame([{
            "model": n, "n_mis": m["n_mis"], "modal_top8": m["modal_top8_mis"],
            "counts": "; ".join(f"{k}={v}" for k, v in
                               sorted(m["top8_counts_mis"].items(),
                                      key=lambda kv: -kv[1])),
        } for n, m in models_out.items()])),
        "",
        "## 6. Criteria and branches",
        "",
    ]
    for k, v in verdicts.items():
        lines.append(f"- **{k}**({v['pass']}/{v['n']}):{v['text']}")
    lines += [
        "",
        "## 7. Reproduction",
        "",
        "```bash",
        "py -B e9c_local_shap.py --full",
        "py -B e9c_local_shap.py --smoke",
        "```",
        "",
        "Per-sample details in `E9c_local_shap.csv`; misclassified top-10 in "
        "`E9c_misclassified_top.csv`.",
    ]
    with open(out("E9c_report.md", tag), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    log(f"done | {time.time() - t0:.1f}s | L1={n_l1}/{len(ev)} L2={n_l2}/{len(ev)} "
        f"L3={n_l3}/{len(ev)} L4={n_l4}/{len(ev)}")
    for k, v in verdicts.items():
        log(f"  {k}: {v['text']}")
    return payload


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="E9c local SHAP and misclassified sample attribution")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--full", action="store_true")
    p.add_argument("--n-explain", type=int, default=None)
    p.add_argument("--n-bg", type=int, default=64)
    p.add_argument("--nsamples", type=int, default=32)
    p.add_argument("--group", type=int, default=16)
    p.add_argument("--b-boot", type=int, default=None)
    a = p.parse_args(argv)
    if a.smoke:
        a.tag = "_smoke"
        a.n_explain = a.n_explain or 48
        a.b_boot = a.b_boot or 200
    else:
        a.tag = ""
        a.n_explain = a.n_explain or 384
        a.b_boot = a.b_boot or 1000
    return a


if __name__ == "__main__":
    main(parse_args())
