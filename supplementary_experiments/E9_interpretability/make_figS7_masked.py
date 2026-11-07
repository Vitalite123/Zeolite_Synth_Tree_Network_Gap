# -*- coding: utf-8 -*-
"""Redraws the E9 consistency figure (Paper2 SI figS7); the right panel no longer annotates nan.

Data sources (no computation rerun):
  - E9_category_summary_all.csv          left panel: 3-category importance shares
  - E9_consistency_rho_vs_permutation.csv right panel: rho (method vs permutation)
Missing cell categorization:
  - "arch": structurally absent (trees have no attention; networks have no gain) -> marked as "—"
  - "na"  : not computed within the E9 preregistered scope (shap for LeNet-5/Transformer/TabNet;
            TabNet has no permutation reference) -> gray cell "n/a"
Output: Paper2/SI/figS7_e9_consistency.png (overwrite)
"""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = __file__.rsplit("\\", 1)[0] if "\\" in __file__ else None
import os

E9_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(E9_DIR, "..", "..", "Paper2", "SI", "figS7_e9_consistency.png")

cat = pd.read_csv(os.path.join(E9_DIR, "E9_category_summary_all.csv"))
rho = pd.read_csv(os.path.join(E9_DIR, "E9_consistency_rho_vs_permutation.csv"))

CATEGORIES = ["Composition", "OSDA", "Synthesis Conditions"]
LABELS = ["Composition", "OSDA", "Synthesis"]

# ---- Left panel: category-share heatmap ----
pivot = cat.pivot_table(index=["model", "method"], columns="category",
                        values="importance_ratio_pct", aggfunc="first")
pivot = pivot.reindex(columns=CATEGORIES)

# ---- Right panel: rho matrix and missing-category classification ----
rho = rho.set_index("model")
methods = list(rho.columns)  # attention, gain, shap
arch_impossible = {("XGBoost-104", "attention")}  # trees have no attention
# gain is tree-model-specific
for m in rho.index:
    if m != "XGBoost-104":
        arch_impossible.add((m, "gain"))

def cell_kind(model, method):
    v = rho.loc[model, method]
    if pd.notna(v):
        return "value", v
    if (model, method) in arch_impossible:
        return "arch", None
    return "na", None

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

im0 = axes[0].imshow(pivot.to_numpy(dtype=float), cmap="YlGnBu", aspect="auto")
axes[0].set_xticks(range(len(CATEGORIES)))
axes[0].set_xticklabels(LABELS, fontsize=9)
axes[0].set_yticks(range(len(pivot)))
axes[0].set_yticklabels([f"{m} / {k}" for m, k in pivot.index], fontsize=8)
axes[0].set_title("Category importance share (%)")
for i in range(len(pivot)):
    for j in range(len(CATEGORIES)):
        v = pivot.iloc[i, j]
        axes[0].text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7)
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

mat = np.full((len(rho), len(methods)), np.nan)
kinds = {}
for i, m in enumerate(rho.index):
    for j, meth in enumerate(methods):
        kind, v = cell_kind(m, meth)
        kinds[(i, j)] = kind
        if kind == "value":
            mat[i, j] = v

im1 = axes[1].imshow(mat, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
axes[1].set_xticks(range(len(methods)))
axes[1].set_xticklabels([c.replace("_", "\n") for c in methods], fontsize=9)
axes[1].set_yticks(range(len(rho)))
axes[1].set_yticklabels(rho.index, fontsize=9)
axes[1].set_title("Spearman rho vs permutation (category ordering)")
for i in range(len(rho)):
    for j in range(len(methods)):
        kind = kinds[(i, j)]
        if kind == "value":
            axes[1].text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)
        elif kind == "arch":
            axes[1].add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                            hatch="///", edgecolor="0.6", lw=0))
            axes[1].text(j, i, "—", ha="center", va="center", fontsize=8, color="0.35")
        else:
            axes[1].add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                            facecolor="0.92", edgecolor="w", lw=2))
            axes[1].text(j, i, "n/a", ha="center", va="center", fontsize=7, color="0.4")
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

fig.text(0.01, 0.015,
         "—: not applicable to the architecture (hatched);  "
         "n/a: not computed in E9 (SHAP restricted to the three primary models; "
         "TabNet has no permutation reference).",
         fontsize=7.5, color="0.25", ha="left")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(os.path.abspath(OUT), dpi=150)
print("saved:", os.path.abspath(OUT))
