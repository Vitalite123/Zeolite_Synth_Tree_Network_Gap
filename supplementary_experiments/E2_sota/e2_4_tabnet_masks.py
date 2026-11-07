"""
E2.4 TabNet feature-mask export (data source for report appendix Fig. S7)
================================================================
Input: e2_2_tabnet_tuned.zip (E2.2 best-config retrained model)
Outputs:
  tabnet_masks_feature.csv   per-feature attention importance (104 rows, descending by importance)
  tabnet_masks_category.csv  aggregated by the 8 feature categories (consistent with report §2.3 / §5.3)

Method: TabNetClassifier.explain(X_test, normalize=True) returns each sample's
feature-level explanation matrix M_explain (n_samples, 104); averaging over test samples gives the global
importance, normalized so Σ=1. The mask is an architecture-intrinsic quantity containing no label information, and can be directly
cross-validation of §5.3 via permutation importance.
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e2_common import (HERE, DEVICE, SEED, FEATURES_104,
                       make_logger, load_data, set_seed)

LOGGER = make_logger("E2_4", "e2_4_run.log")
log = LOGGER.info

MODEL_PATH = os.path.join(HERE, "e2_2_tabnet_tuned.zip")

# 8-category feature grouping: item-by-item correspondence with the report's 104-dim scheme (45+3+4+2+2+5+33+10=104)
CATEGORY_SLICES = [
    ("Element_Composition", 0, 45),
    ("OSDA_Index", 45, 48),
    ("Synthesis_Condition", 48, 52),
    ("Aging_Condition", 52, 54),
    ("pH_Condition", 54, 56),
    ("Gel_Ratio", 56, 61),
    ("OSDA_Descriptor", 61, 94),
    ("OSDA_Aggregate", 94, 104),
]


def build_category_map():
    """Returns a class-name list matching len(FEATURES_104) and verifies slice integrity."""
    cats = [None] * len(FEATURES_104)
    for name, lo, hi in CATEGORY_SLICES:
        for i in range(lo, hi):
            if cats[i] is not None:
                raise RuntimeError(f"feature index {i} grouped more than once")
            cats[i] = name
    if any(c is None for c in cats):
        raise RuntimeError("ungrouped features exist")
    if len(cats) != 104:
        raise RuntimeError(f"feature count {len(cats)} != 104")
    return cats


def main():
    set_seed(SEED)
    log("=" * 64)
    log(f"[E2.4] TabNet feature mask export | device={DEVICE}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"missing E2.2 model: {MODEL_PATH}")

    data = load_data(LOGGER)
    X_test, y_test = data["X_test"], data["y_test"]

    clf = TabNetClassifier(device_name=str(DEVICE), verbose=0, seed=SEED)
    clf.load_model(MODEL_PATH)
    clf.batch_size = 256  # 2GB VRAM; the default 1024 raises the explain forward-pass peak
    log(f"model loaded: {MODEL_PATH}")

    # Consistency check: the model the masks are exported from must be the same weights as E2.2's summary
    proba = clf.predict_proba(X_test)
    acc = float((proba.argmax(axis=1) == y_test).mean())
    log(f"test set recheck Acc={acc:.4f} (E2.2 summary value 0.5661)")

    log("exporting feature masks (explain, normalize=True) ...")
    m_explain, _ = clf.explain(X_test, normalize=True)  # (n_samples, 104)
    if m_explain.shape[1] != 104:
        raise RuntimeError(f"mask dimension {m_explain.shape[1]} != 104")
    imp = m_explain.mean(axis=0)
    imp = imp / imp.sum()

    cats = build_category_map()
    feat_df = pd.DataFrame({
        "feature": FEATURES_104,
        "category": cats,
        "mask_importance": imp,
    }).sort_values("mask_importance", ascending=False, ignore_index=True)
    feat_df["rank"] = np.arange(1, len(feat_df) + 1)
    feat_df = feat_df[["rank", "feature", "category", "mask_importance"]]
    feat_path = os.path.join(HERE, "tabnet_masks_feature.csv")
    feat_df.to_csv(feat_path, index=False, encoding="utf-8")

    cat_df = (feat_df.groupby("category", sort=False)["mask_importance"]
              .agg(["count", "sum", "mean"])
              .rename(columns={"count": "n_features", "sum": "importance_sum",
                               "mean": "importance_mean_per_feature"})
              .reset_index())
    # Ordered as in report section 5.3's display
    order = [c[0] for c in CATEGORY_SLICES]
    cat_df["_order"] = cat_df["category"].map({c: i for i, c in enumerate(order)})
    cat_df = cat_df.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    cat_df["importance_ratio_pct"] = cat_df["importance_sum"] * 100
    cat_path = os.path.join(HERE, "tabnet_masks_category.csv")
    cat_df.to_csv(cat_path, index=False, encoding="utf-8")

    log(f"written: {feat_path} ({len(feat_df)} rows)")
    log(f"written: {cat_path} ({len(cat_df)} rows)")
    log("-" * 64)
    log("class aggregation (Σ=1):")
    for _, r in cat_df.iterrows():
        log(f"  {r['category']:<22} n={int(r['n_features']):>3} "
            f"sum={r['importance_sum']:.4f} ({r['importance_ratio_pct']:.2f}%)")
    log("-" * 64)
    log("feature Top-10:")
    for _, r in feat_df.head(10).iterrows():
        log(f"  #{int(r['rank']):>2} {r['feature']:<28} "
            f"{r['category']:<22} {r['mask_importance']:.5f}")
    log("complete")


if __name__ == "__main__":
    main()
