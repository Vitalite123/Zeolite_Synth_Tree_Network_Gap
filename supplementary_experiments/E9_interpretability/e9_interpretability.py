"""
E9 joint SHAP x attention interpretability (preregistered §9.2)
================================================================
Companion documents: CBUTransfer/DeepLearning/DeepLearning_Supplementary_Plan.md §9.2 (E9)

Purpose
----
answer whether the §5.3 attribution conclusion can be cross-checked by a second independent method:
  1) SHAP (TreeSHAP / GradientSHAP) feature contributions, per
     after aggregating into Composition / OSDA / Synthesis Conditions,
     whether the class ranking agrees with permutation importance (accuracy-drop);
  2) Attention (BiGRU attention, FT-Transformer CLS->token, TabNet mask)
     whether it points to the same class as the two types of evidence above.

Protocol (pre-registered, not adjusted based on results)
--------------------------
- Data: data_processed/ZEOSYN_104_struct_cache.npz (same source as E1/E2/E4);
- Explanation set: test stratified sampling n=384 (smoke 48), seed=42; background = 64 train samples;
- SHAP:
    * XGBoost-104: exact TreeSHAP(Booster.predict(pred_contribs=True)),
      take predicted-class contributions;
    * BiGRU / FT-Transformer: GradientSHAP / expected gradients
      (Erion et al. 2020), implemented with batch interpolation; shap.GradientExplainer is also used
      perform the fidelity check on 8 samples (per-sample feature-level Spearman rho);
- Permutation importance: shuffle each feature within the explanation set (n_repeats=3); metric is the Acc drop;
- Attention: BiGRU 26-position weights averaged into 4 features; FTT takes per-layer
      CLS->feature attention (heads averaged), further averaged over layers/samples;
      TabNet uses the masks exported by e2_4 (architecture-endogenous attention);
- Category rules: reused verbatim from FeatureImportance/create_category_summary_table.py
      's classify_feature_detailed (OSDA first, then cryst/aging/seed/rotation,
      the rest Composition);
- Consistency: Spearman rho of the 3-category importance vectors (3 points, discrete values);
- Acceptance: for models with both SHAP and permutation importance (XGBoost/BiGRU/FTT),
      if the number of models with rho > 0.8 is >= 2/3, §5.3 is upgraded to “dual-method validation”;
      otherwise discuss sources of the difference in §5.4.

Artifacts (this directory)
--------------
E9_report.md / E9_summary.json / E9_consistency_matrix.csv /
shap_category_summary.csv / shap_feature_summary.csv /
perm_category_summary.csv / perm_feature_summary.csv /
attention_category_summary.csv / attention_feature_summary.csv /
E9_metrics.png / e9_run.log
smoke artifacts carry the _smoke suffix.
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
import traceback
from collections import Counter

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.preprocessing import LabelEncoder, StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
SUP = os.path.abspath(os.path.join(HERE, ".."))
E1_DIR = os.path.join(SUP, "E1_tuning")
E2_DIR = os.path.join(SUP, "E2_sota")
for _p in (E1_DIR, E2_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from e2_common import FEATURES_104, SEED, N_CLASS_EXPECT  # noqa: E402

ROOT = os.path.abspath(os.path.join(SUP, ".."))  # Git/ (repo root holding data_processed/)
DATA_DIR = os.path.join(ROOT, "data_processed")
CACHE_PATH = os.path.join(DATA_DIR, "ZEOSYN_104_struct_cache.npz")
XGB_PATH = os.path.join(E1_DIR, "tuned_model.json")
BIGRU_PATH = os.path.join(E1_DIR, "E1_2_tuned.pt")
LENET_PATH = os.path.join(E1_DIR, "E1_1_tuned.pt")
TRANS_PATH = os.path.join(E1_DIR, "E1_3_tuned.pt")
FTT_PATH = os.path.join(E2_DIR, "e2_1_ftt_tuned.pt")
TABNET_FEAT_CSV = os.path.join(E2_DIR, "tabnet_masks_feature.csv")
XGB_GAIN_CSV = os.path.abspath(os.path.join(
    ROOT, "FeatureImportance", "baseline104_xgboost_feature_importance.csv"))

CATEGORIES = ["Composition", "OSDA", "Synthesis Conditions"]

LOG = None


def make_logger(tag):
    lg = logging.getLogger("E9")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
        for h in (logging.FileHandler(os.path.join(HERE, f"e9_run{tag}.log"),
                                      encoding="utf-8"),
                  logging.StreamHandler()):
            h.setFormatter(fmt)
            lg.addHandler(h)
    return lg


def out(name, tag):
    stem, ext = os.path.splitext(name)
    return os.path.join(HERE, f"{stem}{tag}{ext}")


# ==================== Feature three-way classification (verbatim-identical to FeatureImportance) ====================
def classify_feature_detailed(feature_name):
    feature_lower = feature_name.lower()
    if 'osda' in feature_lower or 'sda' in feature_lower:
        return 'OSDA'
    synthesis_conditions = ['cryst', 'aging', 'seed', 'rotation']
    if any(cond in feature_lower for cond in synthesis_conditions):
        return 'Synthesis Conditions'
    elements = ['si', 'al', 'p', 'na', 'k', 'li', 'ca', 'mg', 'ba', 'sr',
                'cs', 'rb', 'ti', 'ge', 'sn', 'zn', 'fe', 'cu', 'co', 'ni',
                'cr', 'mn', 'b', 'ga', 'in', 'f', 'cl', 'br', 'i']
    gel_ratios = ['gel_', 'sio2', 'al2o3', 'p2o5']
    acidity = ['oh_t', 'acid', 'ph', 'h+', 'h2o_t']
    if any(elem in feature_lower for elem in elements):
        return 'Composition'
    if any(ratio in feature_lower for ratio in gel_ratios):
        return 'Composition'
    if any(acid in feature_lower for acid in acidity):
        return 'Composition'
    return 'Composition'


CATS = [classify_feature_detailed(f) for f in FEATURES_104]
CAT_COUNTS = Counter(CATS)
EXPECTED_COUNTS = {"Composition": 52, "OSDA": 46, "Synthesis Conditions": 6}


# ==================== Data ====================
def load_arrays():
    if not os.path.exists(CACHE_PATH):
        raise FileNotFoundError(f"missing data cache: {CACHE_PATH}")
    z = np.load(CACHE_PATH)
    X_train = z["X_train"].astype(np.float32)
    X_val = z["X_val"].astype(np.float32)
    X_test = z["X_test"].astype(np.float32)
    y_train_raw, y_val_raw, y_test_raw = z["y_train"], z["y_val"], z["y_test"]

    le = LabelEncoder().fit(y_train_raw)
    y_train = le.transform(y_train_raw)
    y_val = le.transform(y_val_raw)
    y_test = le.transform(y_test_raw)
    if len(le.classes_) != N_CLASS_EXPECT:
        raise RuntimeError(f"class count {len(le.classes_)} != {N_CLASS_EXPECT}")
    if X_train.shape[1] != len(FEATURES_104):
        raise RuntimeError(f"feature count {X_train.shape[1]} != {len(FEATURES_104)}")

    scaler = StandardScaler().fit(X_train)
    return dict(
        X_train_raw=X_train, X_val_raw=X_val, X_test_raw=X_test,
        X_train_s=scaler.transform(X_train).astype(np.float32),
        X_val_s=scaler.transform(X_val).astype(np.float32),
        X_test_s=scaler.transform(X_test).astype(np.float32),
        y_train=y_train, y_val=y_val, y_test=y_test,
        n_classes=len(le.classes_),
    )


def stratified_sample(y, n, seed):
    """Round-robin sampling by class: ensures long-tail classes are covered, with an upper bound on sample count."""
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    pools = {int(c): rng.permutation(np.where(y == c)[0]).tolist() for c in classes}
    chosen = []
    while len(chosen) < n:
        progressed = False
        for c in classes:
            if pools[int(c)] and len(chosen) < n:
                chosen.append(pools[int(c)].pop())
                progressed = True
        if not progressed:
            break
    return np.array(sorted(chosen), dtype=np.int64)


# ==================== Model loading ====================
class BiGRUAttention(nn.Module):
    """Isomorphic to E1.2 BiGRU_Zeolite; forward additionally returns 26-position attention weights."""

    def __init__(self, num_classes, hidden_size, num_layers, dropout, attention_dim):
        super().__init__()
        self.seq_len = 26
        self.feature_per_step = 4
        self.feature_embedding = nn.Linear(self.feature_per_step, hidden_size // 2)
        self.gru = nn.GRU(input_size=hidden_size // 2, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True, bidirectional=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), nn.Tanh(),
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
        b = x.size(0)
        x = x.view(b, self.seq_len, self.feature_per_step)
        x = self.feature_embedding(x)
        g, _ = self.gru(x)
        w = torch.softmax(self.attention(g), dim=1)  # (B, 26, 1)
        ctx = torch.sum(w * g, dim=1)
        out = self.fc1(ctx)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        return out, w.squeeze(-1)


class LogitsWrap(nn.Module):
    """Wraps any forward (including tuple returns / dual arguments) into a single-argument logits module usable by shap."""

    def __init__(self, model, forward_fn):
        super().__init__()
        self.model = model
        self.forward_fn = forward_fn

    def forward(self, x):
        return self.forward_fn(self.model, x)


def _load_torch(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def build_model(kind):
    """Returns (model, forward_logits, uses_scaled_features)"""
    if kind == "xgb":
        import xgboost as xgb
        booster = xgb.Booster(model_file=XGB_PATH)
        booster.set_param({"nthread": 4})

        def forward(x, _booster=booster):
            dmat = xgb.DMatrix(x, feature_names=list(FEATURES_104))
            return _booster.predict(dmat)

        return booster, forward, False

    if kind == "bigru":
        ck = _load_torch(BIGRU_PATH)
        cfg = ck["config"]
        model = BiGRUAttention(N_CLASS_EXPECT, cfg["hidden_size"], cfg["num_layers"],
                               cfg["dropout_rate"], cfg["attention_dim"])
        model.load_state_dict(ck["model_state_dict"])
        model.to(DEVICE).eval()
        return model, lambda m, x: m(x)[0], True

    if kind == "ftt":
        from e2_1_ft_transformer import build_ftt
        ck = _load_torch(FTT_PATH)
        model = build_ftt(ck["params"], N_CLASS_EXPECT)
        model.load_state_dict(ck["state_dict"])
        model.to(DEVICE).eval()
        return model, lambda m, x: m(x, None), True

    if kind == "lenet":
        from e1_1_lenet5 import build_model as build_lenet
        ck = _load_torch(LENET_PATH)
        model = build_lenet(ck["config"], N_CLASS_EXPECT)
        model.load_state_dict(ck["model_state_dict"])
        model.to(DEVICE).eval()
        return model, lambda m, x: m(x), True

    if kind == "transformer":
        from e1_3_transformer import build_model as build_trans
        ck = _load_torch(TRANS_PATH)
        model = build_trans(ck["config"], N_CLASS_EXPECT)
        model.load_state_dict(ck["model_state_dict"])
        model.to(DEVICE).eval()
        return model, lambda m, x: m(x), True

    raise ValueError(kind)


def torch_proba(model, forward, X, batch=512):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.as_tensor(X[i:i + batch], dtype=torch.float32, device=DEVICE)
            logits = forward(model, xb)
            outs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(outs, axis=0)


# ==================== Permutation importance ====================
def permutation_importance(predict, X, y, n_repeats, seed):
    n, n_features = X.shape
    base_acc = float((predict(X) == y).mean())
    rng = np.random.RandomState(seed)
    raw = np.zeros((n_features, n_repeats), dtype=np.float64)
    for j in range(n_features):
        for r in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = Xp[rng.permutation(n), j]
            raw[j, r] = base_acc - float((predict(Xp) == y).mean())
    return raw.mean(axis=1), raw, base_acc


# ==================== SHAP ====================
def tree_shap_xgb(model, forward, X_raw, pred):
    import xgboost as xgb
    dmat = xgb.DMatrix(X_raw, feature_names=list(FEATURES_104))
    contribs = model.predict(dmat, pred_contribs=True)
    n = len(X_raw)
    if contribs.ndim == 2:
        contribs = contribs.reshape(n, N_CLASS_EXPECT, -1)
    if contribs.shape[0] != n or contribs.shape[1] != N_CLASS_EXPECT:
        raise RuntimeError(f"TreeSHAP shape invalid: {contribs.shape}")
    if contribs.shape[2] != len(FEATURES_104) + 1:
        raise RuntimeError(f"TreeSHAP feature dim invalid: {contribs.shape[2]}")
    return contribs[np.arange(n), pred, :len(FEATURES_104)]


def expected_gradients(model, forward, X_eval, bg, nsamples, seed, group=16):
    """GradientSHAP / expected gradients (predicted class), implemented with batched interpolation.

    attr_i = E_{x'~bg, alpha~U(0,1)}[(x_i - x'_i) * d f_c(x' + alpha(x-x')) / d x_i]
    """
    model.eval()
    n, n_features = X_eval.shape
    bg_t = torch.as_tensor(bg, dtype=torch.float32, device=DEVICE)
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(seed)
    attr = np.zeros((n, n_features), dtype=np.float64)
    pred = np.zeros(n, dtype=np.int64)
    # In eval mode cuDNN RNN does not allow backward; disable cuDNN so GRU uses the native implementation
    with torch.backends.cudnn.flags(enabled=False):
        for start in range(0, n, group):
            end = min(start + group, n)
            e = end - start
            xe = torch.as_tensor(X_eval[start:end], dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                p = forward(model, xe).argmax(dim=1)
            pred[start:end] = p.cpu().numpy()
            bidx = torch.randint(0, bg_t.shape[0], (e, nsamples),
                                 generator=gen, device=DEVICE)
            xbg = bg_t[bidx]  # (E, S, F)
            alpha = torch.rand((e, nsamples, 1), generator=gen, device=DEVICE)
            xint = (xbg + alpha * (xe.unsqueeze(1) - xbg)).reshape(e * nsamples,
                                                                   n_features)
            xint.requires_grad_(True)
            logits = forward(model, xint)
            target = p.repeat_interleave(nsamples)
            sel = logits.gather(1, target.unsqueeze(1)).sum()
            grad = torch.autograd.grad(sel, xint)[0]
            a = ((xe.unsqueeze(1) - xbg) * grad.view(e, nsamples, n_features)).mean(dim=1)
            attr[start:end] = a.detach().cpu().numpy()
    return attr, pred


def shap_fidelity(model, forward, kind, X_cal, bg, nsamples, seed):
    """Compared sample-by-sample against shap.GradientExplainer (feature-level Spearman).

    the official explainer and this implementation both use expected gradients, but the sampling paths differ:
      * ranked_outputs=1 explains only the argmax class, aligned with the “predicted-class contribution” convention,
        and avoid the 231x overhead of the default 231 classes;
      * rseed is fixed to ensure reproducibility;
      * In eval mode cuDNN RNN does not allow backward; cuDNN must be disabled.
    """
    import shap
    our, pred = expected_gradients(model, forward, X_cal, bg, nsamples, seed,
                                   group=min(8, len(X_cal)))
    model.eval()
    wrapper = LogitsWrap(model, forward).to(DEVICE)
    bg_t = torch.as_tensor(bg, dtype=torch.float32, device=DEVICE)
    x_t = torch.as_tensor(X_cal, dtype=torch.float32, device=DEVICE)
    with torch.backends.cudnn.flags(enabled=False):
        expl = shap.GradientExplainer(wrapper, bg_t)
        sv, ranks = expl.shap_values(x_t, nsamples=nsamples, ranked_outputs=1,
                                     output_rank_order="max", rseed=seed)
    arr = np.asarray(sv)
    if arr.ndim == 3:
        arr = arr[..., 0]                      # (N, F, 1) -> (N, F)
    if arr.shape != X_cal.shape:
        raise RuntimeError(f"shap.GradientExplainer output shape invalid: {arr.shape}")
    ranks = ranks.detach().cpu().numpy().reshape(-1) if torch.is_tensor(ranks) \
        else np.asarray(ranks).reshape(-1)
    if not np.array_equal(ranks, pred):
        raise RuntimeError("shap.GradientExplainer ranked_outputs inconsistent with predicted class")
    rhos = []
    for i in range(len(X_cal)):
        r = spearmanr(our[i], arr[i]).statistic
        if np.isfinite(r):
            rhos.append(float(r))
    return float(np.mean(rhos)) if rhos else float("nan"), our, pred


# ==================== Attention ====================
def bigru_attention(model, X, batch=512):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.as_tensor(X[i:i + batch], dtype=torch.float32, device=DEVICE)
            _, w = model(xb)  # (B, 26)
            w104 = (w / 4.0).unsqueeze(-1).expand(-1, -1, 4).reshape(xb.size(0), 104)
            outs.append(w104.cpu().numpy())
    attn = np.concatenate(outs, axis=0).mean(axis=0)
    if abs(float(attn.sum()) - 1.0) > 1e-3:
        raise RuntimeError(f"BiGRU attention not normalized: sum={attn.sum():.6f}")
    return attn


def ftt_attention(model, X, batch=256):
    """Returns (mean CLS->feature attention, mean CLS->CLS self-attention).

    each row of attention_probs is normalized over 105 tokens (1 CLS + 104 features),
    after taking CLS->feature the sum = 1 - P(CLS->CLS), hence the self-attention share is needed as a check.
    """
    blocks = model.transformer.blocks
    buffers = [[] for _ in blocks]
    self_buffers = [[] for _ in blocks]
    hooks = []

    def make_hook(i):
        def hook(mod, inp, output):
            stats = output[1]
            probs = stats["attention_probs"]
            heads = int(getattr(mod, "n_heads"))
            b = probs.shape[0] // heads
            w = probs.view(b, heads, probs.shape[1], probs.shape[2]).mean(dim=1)
            buffers[i].append(w[:, 0, 1:].detach().cpu().numpy())  # CLS -> features
            self_buffers[i].append(w[:, 0, 0].detach().cpu().numpy())  # CLS -> CLS
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
    layer_means = []
    for buf in buffers:
        arr = np.concatenate(buf, axis=0)
        if arr.shape[1] != len(FEATURES_104):
            raise RuntimeError(f"FTT attention feature dim invalid: {arr.shape}")
        layer_means.append(arr.mean(axis=0))
    attn = np.mean(layer_means, axis=0)
    self_mass = float(np.mean([np.concatenate(b).mean() for b in self_buffers]))
    if abs(float(attn.sum()) + self_mass - 1.0) > 1e-2:
        raise RuntimeError(f"FTT attention not normalized: sum={attn.sum():.6f} "
                           f"self={self_mass:.6f}")
    if self_mass > 0.1:
        raise RuntimeError(f"FTT CLS self-attention share abnormally high: {self_mass:.4f}")
    return attn, self_mass


# ==================== Category aggregation and consistency ====================
def category_rows(model_name, method, imp):
    imp = np.asarray(imp, dtype=np.float64)
    pos = np.clip(imp, 0.0, None)
    total = float(pos.sum())
    rows = []
    for cat in CATEGORIES:
        mask = np.array([c == cat for c in CATS])
        s = float(pos[mask].sum())
        rows.append({
            "model": model_name, "method": method, "category": cat,
            "n_features": int(mask.sum()), "importance_sum": s,
            "importance_ratio_pct": (s / total * 100.0) if total > 0 else 0.0,
            "mean_importance_per_feature": s / max(int(mask.sum()), 1),
            "top_feature": FEATURES_104[int(np.argmax(np.where(mask, pos, -1.0)))],
        })
    return rows


def feature_rows(model_name, method, imp):
    imp = np.asarray(imp, dtype=np.float64)
    pos = np.clip(imp, 0.0, None)
    total = float(pos.sum())
    rows = [{
        "model": model_name, "method": method, "feature": FEATURES_104[j],
        "category": CATS[j], "importance_raw": float(imp[j]),
        "importance_share_pct": (float(pos[j]) / total * 100.0) if total > 0 else 0.0,
    } for j in range(len(FEATURES_104))]
    rows.sort(key=lambda r: -r["importance_raw"])
    for k, r in enumerate(rows, 1):
        r["rank"] = k
    return rows


def cat_pct(rows):
    d = {r["category"]: r["importance_ratio_pct"] for r in rows}
    return np.array([d[c] for c in CATEGORIES], dtype=np.float64)


def rho3(a, b):
    if a is None or b is None:
        return float("nan")
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(spearmanr(a, b).statistic)


def md_table(df, digits=2):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append(("nan" if not np.isfinite(v) else f"{v:.{digits}f}"))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ==================== Main pipeline ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    global LOG, DEVICE
    t0 = time.time()
    tag = args.tag
    LOG = make_logger(tag)
    log = LOG.info
    n_explain = args.n_explain
    n_bg = args.n_bg
    n_repeats = args.n_repeats
    nsamples = args.nsamples
    group = args.group

    log("=" * 72)
    log(f"[E9] SHAP x attention joint interpretability | device={DEVICE} | tag='{tag}'")
    log(f"explain set n={n_explain} | background={n_bg} | perm_repeats={n_repeats} | "
        f"GradientSHAP nsamples={nsamples} | fidelity nsamples={args.fid_nsamples}")

    if dict(CAT_COUNTS) != EXPECTED_COUNTS:
        raise RuntimeError(f"feature three-way category counts invalid: {dict(CAT_COUNTS)} != {EXPECTED_COUNTS}")
    log(f"feature three-way category counts: Composition={CAT_COUNTS['Composition']} "
        f"OSDA={CAT_COUNTS['OSDA']} Synthesis={CAT_COUNTS['Synthesis Conditions']}")

    data = load_arrays()
    y_test = data["y_test"]
    idx = stratified_sample(y_test, n_explain, SEED)
    rng = np.random.RandomState(SEED + 1)
    bg_idx = rng.choice(len(data["y_train"]), n_bg, replace=False)
    log(f"explain set class coverage: {len(np.unique(y_test[idx]))} | background from train")

    Xe_raw = data["X_test_raw"][idx]
    Xe_s = data["X_test_s"][idx]
    ye = y_test[idx]
    bg_raw = data["X_train_raw"][bg_idx]
    bg_s = data["X_train_s"][bg_idx]

    specs = [
        ("XGBoost-104", "xgb", 0.8352896630064369),
        ("BiGRU", "bigru", 0.7933),
        ("FT-Transformer", "ftt", 0.7883),
        ("LeNet-5", "lenet", 0.7558),
        ("Transformer", "transformer", 0.7762),
    ]

    feature_all, cat_all = [], []
    summary = {}   # model -> method -> cat pct
    sanity = []
    fidelity = {}

    def record(model_name, method, imp):
        feature_all.extend(feature_rows(model_name, method, imp))
        rows = category_rows(model_name, method, imp)
        cat_all.extend(rows)
        summary.setdefault(model_name, {})[method] = cat_pct(rows)


    for name, kind, ref_acc in specs:
        log("-" * 72)
        log(f"[{name}] loading checkpoint ...")
        model, forward, scaled = build_model(kind)
        X_full = data["X_test_s"] if scaled else data["X_test_raw"]
        Xe = Xe_s if scaled else Xe_raw
        bg = bg_s if scaled else bg_raw

        proba = forward(X_full) if kind == "xgb" else torch_proba(model, forward, X_full)
        acc = float((proba.argmax(axis=1) == y_test).mean())
        delta = acc - ref_acc
        log(f"full test set Acc={acc:.4f} | recorded value {ref_acc:.4f} | Δ={delta*100:+.2f} pp")
        if abs(delta) > 0.005:
            raise RuntimeError(f"{name} loading fidelity failed: Acc={acc:.4f} vs {ref_acc:.4f}")
        sanity.append({"model": name, "test_acc": acc, "ref_acc": ref_acc,
                       "delta_pp": delta * 100.0})

        pred_eval = proba[idx].argmax(axis=1)

        perm, perm_raw, base_acc = permutation_importance(
            ((lambda X, m=model, f=forward: torch_proba(m, f, X).argmax(1))
             if kind != "xgb" else (lambda X, f=forward: f(X).argmax(1))),
            Xe, ye, n_repeats, SEED)
        log(f"permutation importance complete | base Acc (explain set)={base_acc:.4f} | "
            f"Top-3: " + ", ".join(
                f"{FEATURES_104[j]}={perm[j]:+.4f}" for j in np.argsort(-perm)[:3]))
        record(name, "permutation", perm)
        if kind in ("xgb", "bigru", "ftt"):
            if kind == "xgb":
                shap_signed = tree_shap_xgb(model, forward, Xe, pred_eval)
                shap_imp = np.abs(shap_signed).mean(axis=0)
                log("TreeSHAP complete (exact, predicted-class contributions)")
            else:
                shap_signed, pred_shap = expected_gradients(
                    model, forward, Xe, bg, nsamples, SEED, group=group)
                shap_imp = np.abs(shap_signed).mean(axis=0)
                log(f"GradientSHAP complete | pred agreement={int((pred_shap == pred_eval).sum())}/{len(pred_eval)} "
                    f"| mean |attr|={shap_imp.mean():.4f}")
                if args.validate_shap:
                    cal = np.arange(min(8, len(Xe)))
                    rho, _, _ = shap_fidelity(model, forward, kind, Xe[cal], bg,
                                              args.fid_nsamples, SEED)
                    fidelity[name] = rho
                    log(f"SHAP fidelity (per-sample Spearman vs shap.GradientExplainer, "
                        f"nsamples={args.fid_nsamples}): mean rho={rho:.4f}")
                    if not np.isfinite(rho) or rho < 0.5:
                        raise RuntimeError(f"{name} SHAP fidelity insufficient: rho={rho}")
            record(name, "shap", shap_imp)

        if kind == "bigru":
            attn = bigru_attention(model, Xe)
            record(name, "attention", attn)
            log(f"BiGRU attention complete | sum={attn.sum():.4f}")
        elif kind == "ftt":
            attn, self_mass = ftt_attention(model, Xe)
            record(name, "attention", attn)
            log(f"FTT attention complete | CLS->feature sum={attn.sum():.4f} | "
                f"CLS->CLS={self_mass:.4f}")

        del model
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # ---- TabNet masks (existing e2_4 artifacts, architecture-intrinsic attention) ----
    if os.path.exists(TABNET_FEAT_CSV):
        dfm = pd.read_csv(TABNET_FEAT_CSV)
        imp = dfm.set_index("feature").reindex(FEATURES_104)["mask_importance"].to_numpy()
        if np.isnan(imp).any():
            raise RuntimeError("TabNet mask missing some features")
        record("TabNet", "attention", imp)
        log(f"[TabNet] mask class aggregation complete | sum={imp.sum():.4f}")
    else:
        log(f"[TabNet] skipped: missing {TABNET_FEAT_CSV}")

    # ---- XGBoost gain (existing artifact, supplementary method) ----
    if os.path.exists(XGB_GAIN_CSV):
        dfg = pd.read_csv(XGB_GAIN_CSV)
        imp = dfg.set_index("feature").reindex(FEATURES_104)["importance"].fillna(0).to_numpy()
        record("XGBoost-104", "gain", imp)
        log("[XGBoost-104] gain importance class aggregation complete")
    else:
        log(f"[XGBoost-104] gain skipped: missing {XGB_GAIN_CSV}")

    # ---- Consistency matrix ----
    methods = sorted({m for d in summary.values() for m in d})
    matrix_rows = []
    for name in summary:
        perm = summary[name].get("permutation")
        shap = summary[name].get("shap")
        attn = summary[name].get("attention")
        for method in methods:
            if method not in summary[name]:
                continue
            vec = summary[name][method]
            order = " > ".join([CATEGORIES[i] for i in np.argsort(-vec)])
            matrix_rows.append({
                "model": name,
                "method": method,
                "Composition_pct": float(vec[0]),
                "OSDA_pct": float(vec[1]),
                "Synthesis_Conditions_pct": float(vec[2]),
                "top_category": CATEGORIES[int(np.argmax(vec))],
                "rank_order": order,
                "spearman_vs_permutation": (1.0 if method == "permutation"
                                            else rho3(vec, perm)),
                "spearman_vs_shap": (1.0 if method == "shap" else rho3(vec, shap)),
                "spearman_vs_attention": (1.0 if method == "attention"
                                          else rho3(vec, attn)),
            })
    matrix_df = pd.DataFrame(matrix_rows)
    matrix_df.to_csv(out("E9_consistency_matrix.csv", tag), index=False, encoding="utf-8")

    # ---- Acceptance verdict ----
    evaluable = [n for n in summary if "shap" in summary[n] and "permutation" in summary[n]]
    shap_perm = {}
    for n in evaluable:
        shap_perm[n] = rho3(summary[n]["shap"], summary[n]["permutation"])
    n_pass = sum(1 for r in shap_perm.values() if np.isfinite(r) and r > 0.8)
    verdict_pass = n_pass >= 2
    verdict = ("pass: SHAP and permutation importance class rankings agree (>=2/3 models rho>0.8),"
               "section 5.3 attribution upgraded to dual-method validation" if verdict_pass else
               "Fail: the category rankings of SHAP and permutation importance disagree (<2/3 models rho>0.8), "
               "discuss the source of the difference in section 5.4 ")

    log("=" * 72)
    log(f"acceptance: {verdict}")
    for n, r in shap_perm.items():
        log(f"  {n}: rho(SHAP, permutation)={r:.4f}")

    # ---- Persist CSV ----
    feat_df = pd.DataFrame(feature_all)
    feat_df.to_csv(out("E9_feature_importance_all.csv", tag), index=False,
                   encoding="utf-8")
    shap_feat = feat_df[feat_df["method"] == "shap"]
    perm_feat = feat_df[feat_df["method"] == "permutation"]
    attn_feat = feat_df[feat_df["method"] == "attention"]
    shap_feat.to_csv(out("shap_feature_summary.csv", tag), index=False, encoding="utf-8")
    perm_feat.to_csv(out("perm_feature_summary.csv", tag), index=False, encoding="utf-8")
    attn_feat.to_csv(out("attention_feature_summary.csv", tag), index=False,
                     encoding="utf-8")

    cat_df = pd.DataFrame(cat_all)
    cat_df.to_csv(out("E9_category_summary_all.csv", tag), index=False, encoding="utf-8")
    for method, fname in (("shap", "shap_category_summary.csv"),
                          ("permutation", "perm_category_summary.csv"),
                          ("attention", "attention_category_summary.csv")):
        sub = cat_df[cat_df["method"] == method]
        if len(sub):
            sub.to_csv(out(fname, tag), index=False, encoding="utf-8")

    # ---- Figures: category-share heatmaps + heatmap of rank agreement with permutation ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    pivot = cat_df.pivot_table(index=["model", "method"], columns="category",
                               values="importance_ratio_pct", aggfunc="first")
    pivot = pivot.reindex(columns=CATEGORIES)
    im0 = axes[0].imshow(pivot.to_numpy(dtype=float), cmap="YlGnBu", aspect="auto")
    axes[0].set_xticks(range(len(CATEGORIES)))
    axes[0].set_xticklabels(["Composition", "OSDA", "Synthesis"], fontsize=9)
    axes[0].set_yticks(range(len(pivot)))
    axes[0].set_yticklabels([f"{m} / {k}" for m, k in pivot.index], fontsize=8)
    axes[0].set_title("Category importance share (%)")
    for i in range(len(pivot)):
        for j in range(len(CATEGORIES)):
            v = pivot.iloc[i, j]
            axes[0].text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    rho_rows = []
    for name in summary:
        perm = summary[name].get("permutation")
        row = {"model": name}
        for method in methods:
            if method == "permutation":
                row[method] = 1.0 if perm is not None else np.nan
            elif method in summary[name]:
                row[method] = rho3(summary[name][method], perm)
            else:
                row[method] = np.nan
        rho_rows.append(row)
    rho_df = pd.DataFrame(rho_rows).set_index("model")
    rho_df = rho_df.drop(columns=["permutation"], errors="ignore")
    rho_df.to_csv(out("E9_consistency_rho_vs_permutation.csv", tag), encoding="utf-8")
    rho_mat = rho_df.to_numpy(dtype=float)
    im1 = axes[1].imshow(rho_mat, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    axes[1].set_xticks(range(rho_df.shape[1]))
    axes[1].set_xticklabels([c.replace("_", "\n") for c in rho_df.columns], fontsize=9)
    axes[1].set_yticks(range(len(rho_df)))
    axes[1].set_yticklabels(rho_df.index, fontsize=9)
    axes[1].set_title("Spearman rho vs permutation (category ordering)")
    for i in range(rho_mat.shape[0]):
        for j in range(rho_mat.shape[1]):
            v = rho_mat[i, j]
            axes[1].text(j, i, "nan" if not np.isfinite(v) else f"{v:.2f}",
                         ha="center", va="center", fontsize=8)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out("E9_metrics.png", tag), dpi=150)
    plt.close(fig)

    # ---- Report ----
    sanity_df = pd.DataFrame(sanity)
    cat_disp = cat_df[["model", "method", "category", "n_features",
                       "importance_ratio_pct"]].copy()
    matrix_disp = matrix_df[["model", "method", "Composition_pct", "OSDA_pct",
                             "Synthesis_Conditions_pct", "rank_order"]].copy()
    accept_rows = [{"model": n, "rho_shap_vs_permutation": r,
                    "pass(>0.8)": bool(np.isfinite(r) and r > 0.8)}
                   for n, r in shap_perm.items()]
    accept_df = pd.DataFrame(accept_rows)

    lines = []
    lines.append("# E9 SHAP × attention joint interpretability report\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')} | "
                 f"Companion: DeepLearning_Supplementary_Plan.md §9.2\n")
    lines.append("## 1. Protocol\n")
    lines.append(f"- explain set: test stratified sampling n={len(idx)} (seed={SEED}), covering "
                 f"{len(np.unique(ye))} classes; background={n_bg} train samples.")
    lines.append("- SHAP: XGBoost-104 uses exact TreeSHAP (predicted class); BiGRU / "
                 "FT-Transformer uses GradientSHAP (expected gradients, predicted class), "
                 "Also cross-checked against `shap.GradientExplainer` as a fidelity check. ")
    lines.append(f"- permutation importance: per-feature shuffling within the explain set, n_repeats={n_repeats},"
                 "the metric is test Acc drop (aggregated after truncating negatives to 0). ")
    lines.append("- attention: BiGRU's 26 position weights are evenly split across 4 features; FTT takes CLS→feature "
                 "attention (heads/layer/sample averaged); TabNet uses the E2.4 masks. ")
    lines.append("- class rules align with `FeatureImportance/create_category_summary_table.py` "
                 "verbatim-consistent (Composition 52 / OSDA 46 / Synthesis 6).\n")
    lines.append("## 2. Model loading fidelity (full test set)\n")
    lines.append(md_table(sanity_df, 4) + "\n")
    if fidelity:
        lines.append("SHAP fidelity (per-sample feature-level Spearman against the official GradientExplainer,"
                     f"nsamples={args.fid_nsamples}):"
                     + ",".join(f"{k}={v:.3f}" for k, v in fidelity.items()) + ".\n")
    lines.append("## 3. Class importance shares\n")
    lines.append(md_table(cat_disp, 2) + "\n")
    lines.append("## 4. Consistency matrix\n")
    lines.append(md_table(matrix_disp, 2) + "\n")
    lines.append("Spearman ρ of class ranking vs permutation importance (3 classes):\n")
    lines.append(md_table(accept_df, 4) + "\n")
    lines.append("## 5. Acceptance verdict\n")
    lines.append(f"**{verdict}**\n")
    lines.append("Pre-registered criterion: models with both SHAP and permutation importance (XGBoost/BiGRU/FTT)"
                 "If >=2/3 of class rankings have Spearman rho > 0.8, upgrade to dual-method validation.\n ")
    lines.append("## 6. Evidence boundary\n")
    lines.append("- the explain set is a stratified subsample (not the full test set); sampling noise in class shares is roughly "
                 "±1–2 pp;")
    lines.append("- permutation importance shuffles features within the subset, which breaks feature correlation structure,"
                 "may be over-estimated for highly collinear composition features; ")
    lines.append("- SHAP gives predicted-class contributions (not true label class); attributions for misclassified samples reflect model decisions, not data mechanisms;")
    lines.append("- BiGRU/FTT attention is an architecture-intrinsic quantity, not guaranteed causally related to the output;"
                 "In this checkpoint the FTT rtdl implementation has effective heads=8 (recorded as 4); "
                 "this report parses by the module's actual `n_heads`; ")
    lines.append("- TabNet masks come from existing E2.4 artifacts, not recomputed here.\n")
    lines.append("## 7. Reproduction\n")
    lines.append("```bash\npy -B e9_interpretability.py --full\n"
                 "py -B e9_interpretability.py --smoke\n```\n")
    with open(out("E9_report.md", tag), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # ---- Summary JSON ----
    payload = {
        "exp_id": "E9",
        "tag": tag,
        "seed": SEED,
        "device": str(DEVICE),
        "config": {"n_explain": int(len(idx)), "n_background": n_bg,
                   "perm_repeats": n_repeats, "gradientshap_nsamples": nsamples,
                   "fidelity_nsamples": args.fid_nsamples},
        "category_counts": dict(CAT_COUNTS),
        "sanity": sanity,
        "shap_fidelity_spearman": fidelity,
        "category_summary": cat_all,
        "consistency_matrix": matrix_rows,
        "acceptance": {
            "criterion": "SHAP vs permutation category Spearman rho > 0.8 in >=2/3 models",
            "evaluable_models": evaluable,
            "rho_by_model": shap_perm,
            "n_pass": int(n_pass),
            "n_evaluable": len(evaluable),
            "passed": bool(verdict_pass),
            "verdict": verdict,
        },
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(out("E9_summary.json", tag), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

    log(f"artifacts written: {out('E9_report.md', tag)} etc. | elapsed {time.time()-t0:.0f}s")
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="small-sample quick pipeline check")
    p.add_argument("--full", action="store_true", help="full (default)")
    p.add_argument("--tag", default="", help="artifact suffix (underscore added automatically)")
    p.add_argument("--n-explain", type=int, default=None)
    p.add_argument("--n-bg", type=int, default=None)
    p.add_argument("--n-repeats", type=int, default=None)
    p.add_argument("--nsamples", type=int, default=None)
    p.add_argument("--fid-nsamples", type=int, default=None,
                   help="nsamples for the SHAP fidelity check (must be large enough to converge)")
    p.add_argument("--group", type=int, default=16)
    p.add_argument("--validate-shap", action="store_true", default=None)
    p.add_argument("--no-validate-shap", dest="validate_shap", action="store_false")
    args = p.parse_args(argv)

    if args.smoke:
        args.n_explain = args.n_explain or 48
        args.n_bg = args.n_bg or 16
        args.n_repeats = args.n_repeats or 2
        args.nsamples = args.nsamples or 8
        args.fid_nsamples = args.fid_nsamples or 128
        args.tag = args.tag or "_smoke"
        if args.validate_shap is None:
            args.validate_shap = True
    else:
        args.n_explain = args.n_explain or 384
        args.n_bg = args.n_bg or 64
        args.n_repeats = args.n_repeats or 3
        args.nsamples = args.nsamples or 32
        args.fid_nsamples = args.fid_nsamples or 128
        if args.validate_shap is None:
            args.validate_shap = True
    if args.tag and not args.tag.startswith("_"):
        args.tag = "_" + args.tag
    return args


if __name__ == "__main__":
    _rc = 0
    try:
        _rc = main(parse_args())
    except BaseException:
        traceback.print_exc()
        _rc = 1
    finally:
        try:
            logging.shutdown()
        except Exception:
            pass
        os._exit(_rc)
