"""
E8c: rerun E8 with a new base-model family (GINE-SMILES)
================================================================
Pre-registration in DeepLearning_Supplementary_Plan.md §9.1 [Reopen experiment E8c].

Purpose
----
per the upgraded sole reopen condition of E8/E8b (a new base model family whose architecture/information source both differ from XGBoost/BiGRU/FTT
), adding the GINE graph neural network: input is the OSDA molecular graph (SMILES parsed by RDKit
), not reading the 104-dim tabular features.

Metric conventions (verbatim consistent with pre-registration)
----
- New family: hidden 128, 3-layer GINE, dropout 0.2, AdamW lr 1e-3/wd 1e-5, batch 256,
  max 60 epochs, early stopping patience 10 (Macro-F1 on a 10% inner validation split within the fold), gradient clipping 1.0;
- Folds and pairing: directly reuse fold_id / LabelEncoder encoding / inner_split from E8_oof_ms_seed{s}.npz;
- Main convention: weighted_avg4 (OOF simplex grid with step 0.05 to search weights, foldavg inference);
- Criteria: paired same-seed delta-Acc vs E4 XGBoost-104 with mean >= +0.5 pp and 95% CI lower bound > 0;
  Guardrails: Macro-F1(sup) mean ≥ 71.55%, Top-5 mean ≥ 97.30%;
- Anchors: y_train/fold_id bitwise equal to E8 npz; weighted_avg3 recomputed from E8 npz's
  The 5-seed mean differs from the E8 record (83.26%) by <= 0.1 pp.

Artifacts
----
E8c_report.md / E8c_summary.json / E8c_seeds.csv / E8c_methods.csv /
E8c_gnn_metrics.png / E8c_run.log / E8c_gnn_seed{s}.npz / E8c_graphs_cache.pt
"""

import argparse
import csv
import gc
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import rdchem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GINEConv

HERE = os.path.dirname(os.path.abspath(__file__))
SUP_DIR = os.path.abspath(os.path.join(HERE, ".."))
E2_DIR = os.path.join(SUP_DIR, "E2_sota")
if E2_DIR not in sys.path:
    sys.path.insert(0, E2_DIR)
import e2_common as ec  # noqa: E402

RDLogger.DisableLog("rdApp.*")

# ==================== Constants ====================
N_CLASSES = ec.N_CLASS_EXPECT
OLD_MODELS = ["XGBoost-104", "BiGRU", "FT-Transformer"]
SLUG = {"XGBoost-104": "xgboost", "BiGRU": "bigru", "FT-Transformer": "ftt"}
GNN_NAME = "GINE"
GNN_KEY = "gine"
SEEDS = [42, 123, 777, 2026, 3407]
E4_SEED_CSV = os.path.join(SUP_DIR, "E4_seeds", "seed_repetition_results.csv")

# Hyperparameters frozen (pre-registered, not tuned)
HIDDEN = 128
LAYERS = 3
DROPOUT = 0.2
BATCH = 256
LR = 1e-3
WD = 1e-5
MAX_EPOCHS = 60
PATIENCE = 10
GRAD_CLIP = 1.0

# Graph feature dimension
ATOM_EMB = 32
MAX_ATOM = 100
CONT_DIM = 25
BOND_DIM = 7
PRESENCE = 3

# Criteria / guardrails
DELTA_PP_REQUIRED = 0.5
GUARD_MACRO_F1 = 0.7155
GUARD_TOP5 = 0.9730
ANCHOR_TOL_PP = 0.1          # weighted_avg3 recomputation vs E8 record
ANCHOR_TOL_ACC = 0.2         # fine tolerance for the fully E8-consistent source (unused, kept for audit)

LOG = None


def log(msg):
    if LOG is not None:
        LOG.info(msg)
    print(msg, flush=True)


def _json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def e8_npz(seed):
    return os.path.join(HERE, f"E8_oof_ms_seed{seed}.npz")


def e8_summary(seed):
    return os.path.join(HERE, f"E8_summary_ms_seed{seed}.json")


def gnn_npz(seed):
    return os.path.join(HERE, f"E8c_gnn_seed{seed}.npz")


GRAPH_CACHE = os.path.join(HERE, "E8c_graphs_cache.pt")


# ==================== Graph features ====================
def _atom_feats(atom):
    hyb = {rdchem.HybridizationType.S: 0, rdchem.HybridizationType.SP: 1,
           rdchem.HybridizationType.SP2: 2, rdchem.HybridizationType.SP3: 3}
    atomic = int(min(atom.GetAtomicNum(), MAX_ATOM - 1))
    cont = np.zeros(CONT_DIM, dtype=np.float32)
    cont[int(min(atom.GetDegree(), 5))] = 1.0
    cont[6 + int(np.clip(atom.GetFormalCharge(), -2, 2)) + 2] = 1.0
    cont[11 + int(min(atom.GetTotalNumHs(), 4))] = 1.0
    cont[16 + (1 if atom.GetIsAromatic() else 0)] = 1.0
    cont[18 + (1 if atom.IsInRing() else 0)] = 1.0
    cont[20 + hyb.get(atom.GetHybridization(), 4)] = 1.0
    return atomic, cont


def _bond_feats(bond):
    bt = float(bond.GetBondTypeAsDouble())
    v = np.zeros(BOND_DIM, dtype=np.float32)
    if abs(bt - 1.0) < 1e-6:
        v[0] = 1.0
    elif abs(bt - 2.0) < 1e-6:
        v[1] = 1.0
    elif abs(bt - 3.0) < 1e-6:
        v[2] = 1.0
    elif abs(bt - 1.5) < 1e-6:
        v[3] = 1.0
    else:
        v[4] = 1.0
    v[5] = 1.0 if bond.GetIsConjugated() else 0.0
    v[6] = 1.0 if bond.IsInRing() else 0.0
    return v


def mol_graph(smiles):
    """Returns {'x_atomic','x_cont','edge_index','edge_attr'} or None (on parse failure)."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    xs, cs = [], []
    for atom in mol.GetAtoms():
        a, c = _atom_feats(atom)
        xs.append([a])
        cs.append(c)
    src, dst, eas = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = _bond_feats(bond)
        src += [i, j]
        dst += [j, i]
        eas += [bf, bf]
    return dict(
        x_atomic=np.asarray(xs, dtype=np.int64).reshape(-1),
        x_cont=np.asarray(cs, dtype=np.float32),
        edge_index=np.asarray([src, dst], dtype=np.int64).reshape(2, -1),
        edge_attr=(np.asarray(eas, dtype=np.float32) if eas
                   else np.zeros((0, BOND_DIM), dtype=np.float32)),
    )


def make_sample_data(smiles3, mol_cache):
    """The three OSDA slots are combined into one graph; missing slots are padded with zero atoms (presence=0)."""
    x_at, x_ct, slot, e_src, e_dst, e_attr = [], [], [], [], [], []
    presence = np.zeros(PRESENCE, dtype=np.float32)
    off = 0
    for k, s in enumerate(smiles3):
        g = mol_cache.get(s) if isinstance(s, str) and s.strip() else None
        if g is None:
            x_at.append(np.zeros(1, dtype=np.int64))   # dummy atom for padding_idx=0
            x_ct.append(np.zeros((1, CONT_DIM), dtype=np.float32))
            slot.append(np.array([k], dtype=np.int64))
            off += 1
            continue
        n = g["x_atomic"].shape[0]
        x_at.append(g["x_atomic"])
        x_ct.append(g["x_cont"])
        slot.append(np.full(n, k, dtype=np.int64))
        e_src.append(g["edge_index"][0] + off)
        e_dst.append(g["edge_index"][1] + off)
        e_attr.append(g["edge_attr"])
        presence[k] = 1.0
        off += n
    ei = (np.asarray([np.concatenate(e_src), np.concatenate(e_dst)], dtype=np.int64)
          if e_src else np.zeros((2, 0), dtype=np.int64))
    ea = (np.concatenate(e_attr) if e_attr
          else np.zeros((0, BOND_DIM), dtype=np.float32))
    x_at = torch.from_numpy(np.concatenate(x_at))
    return Data(
        x_atomic=x_at,
        x_cont=torch.from_numpy(np.concatenate(x_ct)),
        edge_index=torch.from_numpy(ei),
        edge_attr=torch.from_numpy(ea),
        slot=torch.from_numpy(np.concatenate(slot)),
        presence=torch.from_numpy(presence.reshape(1, PRESENCE)),
        num_nodes=int(x_at.numel()),   # explicit node count: dummy atoms are often isolated points, avoid inference bias
    )


def build_graph_cache(y_train, y_val, y_test):
    t0 = time.time()
    mol_cache = {}
    out = {}
    for split, y in (("train", y_train), ("val", y_val), ("test", y_test)):
        df = pd.read_excel(
            os.path.join(ec.DATA_DIR, f"ZEOSYN_104_struct_{split}.xlsx"),
            usecols=["osda1 smiles", "osda2 smiles", "osda3 smiles"])
        if len(df) != len(y):
            raise RuntimeError(f"{split} row count {len(df)} != labels {len(y)}")
        data_list = []
        for i in range(len(df)):
            smis = [df.iloc[i, 0], df.iloc[i, 1], df.iloc[i, 2]]
            for s in smis:
                if isinstance(s, str) and s.strip() and s not in mol_cache:
                    mol_cache[s] = mol_graph(s)
            d = make_sample_data(smis, mol_cache)
            d.y = torch.tensor([int(y[i])], dtype=torch.long)
            data_list.append(d)
        out[split] = data_list
        n_osda = int(sum(float(d.presence.sum()) > 0 for d in data_list))
        n_atoms = int(sum(d.x_atomic.numel() for d in data_list))
        log(f"[graph] {split}: {len(data_list)} samples | with OSDA {n_osda} | nodes {n_atoms}")
    torch.save(out, GRAPH_CACHE)
    log(f"[graph] unique molecules {len(mol_cache)} | cache {GRAPH_CACHE} | {time.time()-t0:.1f}s")
    return out


class GraphSubset(torch.utils.data.Dataset):
    def __init__(self, graphs, idx):
        self.graphs = graphs
        self.idx = np.asarray(idx, dtype=np.int64)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return self.graphs[int(self.idx[i])]


# ==================== Models ====================
class GINEClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(MAX_ATOM, ATOM_EMB, padding_idx=0)
        self.proj = nn.Linear(ATOM_EMB + CONT_DIM, HIDDEN)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(LAYERS):
            mlp = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
                                nn.Linear(HIDDEN, HIDDEN))
            self.convs.append(GINEConv(mlp, edge_dim=BOND_DIM))
            self.bns.append(nn.BatchNorm1d(HIDDEN))
        self.drop = nn.Dropout(DROPOUT)
        self.head = nn.Sequential(
            nn.Linear(3 * HIDDEN + PRESENCE, 256), nn.ReLU(),
            nn.Dropout(DROPOUT), nn.Linear(256, N_CLASSES))

    def forward(self, data):
        x = torch.cat([self.emb(data.x_atomic), data.x_cont], dim=-1)
        h = self.proj(x)
        for conv, bn in zip(self.convs, self.bns):
            h = h + self.drop(F.relu(bn(conv(h, data.edge_index, data.edge_attr))))
        b = data.num_graphs
        group = data.batch * PRESENCE + data.slot
        pooled = torch.zeros(b * PRESENCE, HIDDEN, device=h.device, dtype=h.dtype)
        pooled.index_add_(0, group, h)
        cnt = torch.zeros(b * PRESENCE, device=h.device, dtype=h.dtype)
        cnt = cnt.index_add_(0, group, torch.ones_like(h[:, 0]))
        pooled = (pooled / cnt.clamp(min=1.0).unsqueeze(-1)).view(b, PRESENCE, HIDDEN)
        pooled = pooled * data.presence.unsqueeze(-1)
        return self.head(torch.cat([pooled.flatten(1), data.presence], dim=-1))


def macro_f1_support(y, pred):
    present = np.unique(y)
    return float(f1_score(y, pred, labels=present, average="macro", zero_division=0))


def full_metrics(proba, y, strata):
    m = ec.metrics_from_proba(proba, y, N_CLASSES, strata)
    m["macro_f1_support"] = macro_f1_support(y, proba.argmax(axis=1))
    m["n_samples"] = int(len(y))
    return m


@torch.no_grad()
def predict(model, graphs, idx, bs=512):
    model.eval()
    out = np.zeros((len(idx), N_CLASSES), dtype=np.float32)
    loader = PyGDataLoader(GraphSubset(graphs, idx), batch_size=bs, shuffle=False)
    p = 0
    for b in loader:
        b = b.to(ec.DEVICE)
        out[p:p + b.num_graphs] = torch.softmax(model(b), 1).float().cpu().numpy()
        p += b.num_graphs
    return out


def inner_split(idx_tr, y, seed, frac=0.10):
    """Fold-internal split identical to E8 (single-sample classes all stay on the training side)."""
    counts = np.bincount(y[idx_tr], minlength=N_CLASSES)
    single = np.where(counts == 1)[0]
    is_single = np.isin(y[idx_tr], single)
    pool = idx_tr[~is_single]
    keep = idx_tr[is_single]
    a, b = train_test_split(pool, test_size=frac, stratify=y[pool], random_state=seed)
    return np.sort(np.concatenate([a, keep])), np.sort(b)


def train_fold(graphs, y, idx_fit, idx_in, seed, max_epochs, patience, tag=""):
    ec.set_seed(seed)
    model = GINEClassifier().to(ec.DEVICE)
    gen = torch.Generator()
    gen.manual_seed(seed)
    loader = PyGDataLoader(GraphSubset(graphs, idx_fit), batch_size=BATCH,
                           shuffle=True, num_workers=0, generator=gen)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    lossf = nn.CrossEntropyLoss()
    best_f, best_state, best_ep, bad = -1.0, None, 0, 0
    for ep in range(1, max_epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for b in loader:
            b = b.to(ec.DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(b), b.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            tot += float(loss) * b.num_graphs
            n += b.num_graphs
        p_in = predict(model, graphs, idx_in)
        f = macro_f1_support(y[idx_in], p_in.argmax(axis=1))
        mark = ""
        if f > best_f + 1e-6:
            best_f = f
            best_ep = ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
            mark = " *"
        else:
            bad += 1
        log(f"{tag} ep {ep:03d} | loss {tot/max(n,1):.4f} | inner-val MacroF1 {f:.4f}"
            f" | best {best_f:.4f}@{best_ep}{mark}")
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    n_params = int(sum(p.numel() for p in model.parameters()))
    log(f"{tag} early stopping best_ep={best_ep} | inner-val MacroF1={best_f:.4f} | "
        f"params={n_params:,}")
    return model, best_ep, ep, n_params


# ==================== Single-seed main pipeline ====================
def run_seed(seed, smoke=False, force=False):
    suffix = "_smoke" if smoke else ""
    out_npz = gnn_npz(seed) if not smoke else os.path.join(HERE, f"E8c_gnn_smoke_seed{seed}.npz")
    if os.path.exists(out_npz) and not force:
        log(f"[seed {seed}] {out_npz} exists, skipping (use --force to rerun)")
        return
    raw = np.load(ec.CACHE_PATH)
    le = LabelEncoder()
    y_train = le.fit_transform(raw["y_train"]).astype(np.int64)
    y_val = le.transform(raw["y_val"]).astype(np.int64)
    y_test = le.transform(raw["y_test"]).astype(np.int64)
    if len(le.classes_) != N_CLASSES:
        raise RuntimeError(f"class count {len(le.classes_)} != {N_CLASSES}")

    ref = np.load(e8_npz(seed))
    if not np.array_equal(ref["y_train"], y_train):
        raise AssertionError(f"[anchor] seed {seed}: y_train inconsistent with E8 npz")
    fold_id = ref["fold_id"].astype(np.int64)
    if (fold_id < 0).any():
        raise AssertionError(f"[anchor] seed {seed}: fold_id has uncovered samples")

    if os.path.exists(GRAPH_CACHE):
        graphs = torch.load(GRAPH_CACHE, weights_only=False)
    else:
        graphs = build_graph_cache(y_train, y_val, y_test)
    g_tr, g_va, g_te = graphs["train"], graphs["val"], graphs["test"]

    n_folds = 5
    n_tr = len(y_train)
    oof = np.zeros((n_tr, N_CLASSES), dtype=np.float32)
    fold_test, fold_val = [], []
    runs = []
    t_seed = time.time()
    for fi in range(n_folds):
        idx_ho = np.where(fold_id == fi)[0]
        idx_tr = np.where(fold_id != fi)[0]
        idx_fit, idx_in = inner_split(idx_tr, y_train, seed)
        tag = f"[seed {seed}][fold {fi}]"
        log(f"{tag} fit={len(idx_fit)} inner={len(idx_in)} holdout={len(idx_ho)}")
        t0 = time.time()
        model, best_ep, ep_run, n_params = train_fold(
            g_tr, y_train, idx_fit, idx_in, seed,
            max_epochs=(2 if smoke else MAX_EPOCHS),
            patience=(1 if smoke else PATIENCE), tag=tag)
        p_ho = predict(model, g_tr, idx_ho)
        p_te = predict(model, g_te, np.arange(len(y_test)))
        p_va = predict(model, g_va, np.arange(len(y_val)))
        oof[idx_ho] = p_ho
        fold_test.append(p_te)
        fold_val.append(p_va)
        m_ho = macro_f1_support(y_train[idx_ho], p_ho.argmax(axis=1))
        m_te = full_metrics(p_te, y_test, ec.make_strata(y_train, N_CLASSES))
        log(f"{tag} done {time.time()-t0:.1f}s | holdout Acc "
            f"{(p_ho.argmax(1)==y_train[idx_ho]).mean():.4f} MacroF1 {m_ho:.4f} | "
            f"test Acc {m_te['acc']:.4f}")
        runs.append(dict(seed=seed, fold=fi, best_epoch=best_ep, epochs_run=ep_run,
                         n_fit=len(idx_fit), n_inner=len(idx_in),
                         n_holdout=len(idx_ho), n_params=n_params,
                         holdout_acc=round(float((p_ho.argmax(1) == y_train[idx_ho]).mean()), 6),
                         holdout_macro_f1=round(m_ho, 6),
                         test_acc=round(m_te["acc"], 6),
                         test_macro_f1_support=round(m_te["macro_f1_support"], 6),
                         test_top5=round(m_te["top5"], 6),
                         duration_s=round(time.time() - t0, 1)))
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Persist per fold: isomorphic to E8, directly reusable by fusion analysis
    np.savez_compressed(
        out_npz,
        fold_id=ref["fold_id"], y_train=y_train, y_val=y_val, y_test=y_test,
        oof_gine=oof.astype(np.float16),
        test_gine=np.mean(fold_test, axis=0).astype(np.float16),
        fold_test_gine=np.stack(fold_test).astype(np.float16),
        fold_val_gine=np.stack(fold_val).astype(np.float16),
        runs_json=np.array(json.dumps(runs)),
    )
    log(f"[seed {seed}] done {time.time()-t_seed:.1f}s -> {out_npz}")
    return runs


# ==================== Fusion analysis ====================
def simple_average(probs, models):
    return np.mean([probs[m] for m in models], axis=0).astype(np.float32)


def search_weights(probs, y, models, step=0.05):
    """Exactly the same implementation as E8: OOF simplex grid search for Acc-optimal weights."""
    best_w, best_acc = None, -1.0
    n = len(models)
    grid = np.arange(0.0, 1.0 + 1e-9, step)

    def rec(i, remain, cur):
        nonlocal best_w, best_acc
        if i == n - 1:
            w = np.array(cur + [round(remain, 6)], dtype=np.float64)
            p = sum(w[k] * probs[models[k]] for k in range(n))
            acc = float((p.argmax(axis=1) == y).mean())
            if acc > best_acc:
                best_acc, best_w = acc, w
            return
        for w in grid:
            if w > remain + 1e-9:
                break
            rec(i + 1, remain - w, cur + [float(w)])

    rec(0, 1.0, [])
    return best_w, best_acc


def weighted_average(probs, weights, models):
    return sum(float(weights[k]) * probs[models[k]]
               for k in range(len(models))).astype(np.float32)


def lr_stack(oof, y, folds_probs, models, C=1.0, max_iter=2000):
    Xo = np.concatenate([oof[m] for m in models], axis=1).astype(np.float32)
    clf = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs", tol=1e-4)
    clf.fit(Xo, y)

    def expand(p):
        out = np.zeros((len(p), N_CLASSES), dtype=np.float32)
        out[:, clf.classes_] = p
        return out

    outs = []
    for fi in range(len(folds_probs[models[0]])):
        Xf = np.concatenate([folds_probs[m][fi] for m in models], axis=1).astype(np.float32)
        outs.append(expand(clf.predict_proba(Xf)))
    return np.mean(outs, axis=0).astype(np.float32)


def ci95(d):
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    mean = float(d.mean())
    if n < 2:
        return mean, float("nan"), float("nan")
    se = float(d.std(ddof=1) / math.sqrt(n))
    try:
        from scipy import stats
        crit = float(stats.t.ppf(0.975, n - 1))
    except Exception:
        crit = 2.776
    return mean, mean - crit * se, mean + crit * se


def load_e4_per_seed():
    d = pd.read_csv(E4_SEED_CSV)
    d = d[d["model"].astype(str) == "XGBoost-104"]
    return {int(r.seed): float(r.acc) for r in d.itertuples()}


def analyze(seeds):
    raw = np.load(ec.CACHE_PATH)
    le = LabelEncoder()
    y_train = le.fit_transform(raw["y_train"]).astype(np.int64)
    y_val = le.transform(raw["y_val"]).astype(np.int64)
    y_test = le.transform(raw["y_test"]).astype(np.int64)
    strata_tr = ec.make_strata(y_train, N_CLASSES)
    e4 = load_e4_per_seed()

    methods = ["E4_XGBoost-104(full-train)", "GINE_alone_foldavg", "weighted_avg3_anchor",
               "weighted_avg4_primary", "stacking4_foldavg"]
    per_seed = {m: {} for m in methods}
    weights3, weights4, oracle, jacc = {}, {}, {}, {}
    anchors = []
    for s in seeds:
        ref = np.load(e8_npz(s))
        cur = np.load(gnn_npz(s))
        if not np.array_equal(ref["y_train"], cur["y_train"]):
            raise AssertionError(f"[anchor] seed {s}: y_train inconsistent")
        if not np.array_equal(ref["fold_id"], cur["fold_id"]):
            raise AssertionError(f"[anchor] seed {s}: fold_id inconsistent")
        models3 = OLD_MODELS
        models4 = OLD_MODELS + [GNN_NAME]
        probs_oof3 = {m: ref[f"oof_{SLUG[m]}"].astype(np.float32) for m in models3}
        probs_oof4 = dict(probs_oof3)
        probs_oof4[GNN_NAME] = cur["oof_gine"].astype(np.float32)
        folds3 = {m: list(ref[f"fold_test_{SLUG[m]}"].astype(np.float32)) for m in models3}
        folds4 = dict(folds3)
        folds4[GNN_NAME] = list(cur["fold_test_gine"].astype(np.float32))

        w3, _ = search_weights(probs_oof3, y_train, models3)
        w4, oof_acc4 = search_weights(probs_oof4, y_train, models4)
        weights3[s], weights4[s] = [round(float(x), 4) for x in w3], [round(float(x), 4) for x in w4]

        p3 = np.mean([weighted_average({m: folds3[m][fi] for m in models3}, w3, models3)
                      for fi in range(len(folds3[models3[0]]))], axis=0)
        p4 = np.mean([weighted_average({m: folds4[m][fi] for m in models4}, w4, models4)
                      for fi in range(len(folds4[models4[0]]))], axis=0)
        p_gnn = np.mean([folds4[GNN_NAME][fi] for fi in range(len(folds4[GNN_NAME]))], axis=0)
        p_stack4 = lr_stack(probs_oof4, y_train, folds4, models4, C=1.0)

        va = {
            "E4_XGBoost-104(full-train)": None,
            "GINE_alone_foldavg": full_metrics(p_gnn, y_test, strata_tr),
            "weighted_avg3_anchor": full_metrics(p3, y_test, strata_tr),
            "weighted_avg4_primary": full_metrics(p4, y_test, strata_tr),
            "stacking4_foldavg": full_metrics(p_stack4, y_test, strata_tr),
        }
        for m in methods:
            if m == "E4_XGBoost-104(full-train)":
                per_seed[m][s] = dict(acc=e4[s], macro_f1_support=None, weighted_f1=None, top5=None)
            else:
                per_seed[m][s] = dict(acc=float(va[m]["acc"]),
                                      macro_f1_support=float(va[m]["macro_f1_support"]),
                                      weighted_f1=float(va[m]["weighted_f1"]),
                                      top5=float(va[m]["top5"]))

        preds = {m: np.mean([folds4[m][fi] for fi in range(5)], axis=0) for m in models4}
        pred_lab = {m: preds[m].argmax(1) for m in models4}
        oracle[s] = float(np.any([pred_lab[m] == y_test for m in models4], axis=0).mean())
        jacc[s] = {}
        for m in OLD_MODELS:
            a_err = pred_lab[GNN_NAME] != y_test
            b_err = pred_lab[m] != y_test
            jacc[s][m] = round(float((a_err & b_err).sum() / max((a_err | b_err).sum(), 1)), 6)
        anchors.append(dict(seed=s, w3=weights3[s], w4=weights4[s], oof_acc_w4=round(oof_acc4, 6)))

    # ---- Aggregation and paired statistics ----
    agg = {}
    for m in methods:
        accs = np.array([per_seed[m][s]["acc"] for s in seeds], dtype=np.float64)
        agg[m] = dict(acc_mean=float(accs.mean()), acc_std=float(accs.std(ddof=1)),
                      per_seed=[float(x) for x in accs])
        if m != "E4_XGBoost-104(full-train)":
            for k in ("macro_f1_support", "weighted_f1", "top5"):
                vals = np.array([per_seed[m][s][k] for s in seeds], dtype=np.float64)
                agg[m][k] = dict(mean=float(vals.mean()), std=float(vals.std(ddof=1)),
                                 per_seed=[float(x) for x in vals])

    def paired(a, b):
        d = np.array([(per_seed[a][s]["acc"] - (e4[s] if b == "E4_XGBoost-104(full-train)"
                                                else per_seed[b][s]["acc"])) * 100
                      for s in seeds], dtype=np.float64)
        mean, lo, hi = ci95(d)
        return dict(mean_pp=mean, ci95_low=lo, ci95_high=hi,
                    per_seed=[round(float(x), 3) for x in d])

    deltas = {
        "primary_vs_e4": paired("weighted_avg4_primary", "E4_XGBoost-104(full-train)"),
        "gnn_vs_e4": paired("GINE_alone_foldavg", "E4_XGBoost-104(full-train)"),
        "w3_vs_e4": paired("weighted_avg3_anchor", "E4_XGBoost-104(full-train)"),
        "stack4_vs_e4": paired("stacking4_foldavg", "E4_XGBoost-104(full-train)"),
        "primary_vs_w3": paired("weighted_avg4_primary", "weighted_avg3_anchor"),
        "primary_vs_stack4": paired("weighted_avg4_primary", "stacking4_foldavg"),
    }

    # ---- Anchor: recomputed weighted_avg3 vs E8's recorded 83.26% ----
    w3_mean = agg["weighted_avg3_anchor"]["acc_mean"]
    anchor_delta_pp = abs(w3_mean * 100 - 83.26)
    anchor_ok = anchor_delta_pp <= ANCHOR_TOL_PP

    # ---- Guardrails ----
    prim = agg["weighted_avg4_primary"]
    mf1 = prim["macro_f1_support"]
    top5 = prim["top5"]
    guard = dict(
        macro_f1_support_mean=round(mf1["mean"], 6), macro_f1_support_ref=GUARD_MACRO_F1,
        macro_f1_support_pass=bool(mf1["mean"] >= GUARD_MACRO_F1),
        macro_f1_support_seeds_pass=[int(sum(x >= GUARD_MACRO_F1 for x in mf1["per_seed"])), len(seeds)],
        top5_mean=round(top5["mean"], 6), top5_ref=GUARD_TOP5,
        top5_pass=bool(top5["mean"] >= GUARD_TOP5),
        top5_seeds_pass=[int(sum(x >= GUARD_TOP5 for x in top5["per_seed"])), len(seeds)],
    )

    # ---- Verdict (pre-registered three branches) ----
    d_primary = deltas["primary_vs_e4"]
    acc_pass = bool(d_primary["mean_pp"] >= DELTA_PP_REQUIRED and d_primary["ci95_low"] > 0)
    w4_gine_pp = float(np.mean([w[3] for w in weights4.values()]) * 100)
    w4_gine_positive = bool(any(w[3] > 0 for w in weights4.values()))
    improve_pp = deltas["primary_vs_w3"]["mean_pp"]
    if acc_pass:
        branch, branch_text = "new_family_pass", "new-family branch holds: introducing SMILES molecular graphs brings complementarity"
    elif w4_gine_positive and improve_pp >= 0.10:
        branch, branch_text = "directional", "directionally complementary but below threshold: E8 stays closed"
    else:
        branch, branch_text = "none", "new-family branch does not hold: E8 stays closed"
    verdict = dict(
        criterion="weighted_avg4 vs E4 XGBoost-104 paired same-seed ΔAcc mean >= +0.5 pp and 95% CI lower bound > 0",
        primary_method="weighted_avg4_primary",
        anchor_weighted_avg3_mean=round(float(w3_mean), 6),
        anchor_weighted_avg3_ref=0.8326,
        anchor_abs_delta_pp=round(float(anchor_delta_pp), 4),
        anchor_pass=bool(anchor_ok),
        primary=d_primary, acc_pass=acc_pass,
        guardrail=guard,
        gine_weight_mean_pp=round(w4_gine_pp, 3),
        gine_weight_positive=w4_gine_positive,
        primary_minus_w3_pp=round(float(improve_pp), 3),
        branch=branch, branch_text=branch_text,
        oracle=oracle, error_jaccard=jacc, weights=dict(w3=weights3, w4=weights4),
    )

    summary = dict(
        experiment="E8c", date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        seeds=[int(x) for x in seeds],
        new_family=dict(name="GINE-SMILES", hidden=HIDDEN, layers=LAYERS, dropout=DROPOUT,
                        batch=BATCH, lr=LR, wd=WD, max_epochs=MAX_EPOCHS, patience=PATIENCE,
                        info_source="osda1/2/3 SMILES -> RDKit molecular graph (no 104 features)"),
        aggregate=agg, deltas=deltas, verdict=verdict,
    )

    # ---- Persist to disk ----
    with open(os.path.join(HERE, "E8c_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    rows = []
    for m in methods:
        for s in seeds:
            r = dict(method=m, seed=s, **{k: v for k, v in per_seed[m][s].items()})
            rows.append(r)
    with open(os.path.join(HERE, "E8c_seeds.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["method", "seed", "acc", "macro_f1_support",
                                           "weighted_f1", "top5"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    mrows = []
    for m in methods:
        r = dict(method=m, acc_mean=agg[m]["acc_mean"], acc_std=agg[m]["acc_std"])
        for k in ("macro_f1_support", "weighted_f1", "top5"):
            if k in agg[m]:
                r[k] = agg[m][k]["mean"]
                r[f"{k}_std"] = agg[m][k]["std"]
        mrows.append(r)
    with open(os.path.join(HERE, "E8c_methods.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["method", "acc_mean", "acc_std",
                                           "macro_f1_support", "macro_f1_support_std",
                                           "weighted_f1", "weighted_f1_std", "top5", "top5_std"])
        w.writeheader()
        for r in mrows:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})

    write_report(summary, agg, deltas, guard, anchors, e4)
    plot_metrics(agg, deltas, methods, seeds)
    log(f"[analyze] E8c summary written | branch={branch}")
    return summary


def write_report(summary, agg, deltas, guard, anchors, e4):
    v = summary["verdict"]
    d = deltas["primary_vs_e4"]
    dg = deltas["gnn_vs_e4"]
    d3 = deltas["w3_vs_e4"]
    lines = []
    lines.append("# E8c: new base model family (GINE-SMILES) reopen experiment (5-seed paired)\n")
    lines.append(f"- Date: {summary['date']} | Seeds: {summary['seeds']} | "
                 f"Primary metric: weighted_avg4_primary (foldavg)")
    lines.append(f"- Criterion: {v['criterion']}\n")
    lines.append("## 1. Conclusion\n")
    if v["branch"] == "new_family_pass":
        lines.append("- **New base model family branch holds: E8 dimension line reopens**")
    elif v["branch"] == "directional":
        lines.append("- **Directional complementarity but below threshold: E8 dimension line stays closed**")
    else:
        lines.append("- **New base model family branch does not hold: E8 dimension line stays closed**")
    lines.append(f"- {v['branch_text']}")
    lines.append(f"- primary-metric ΔAcc vs E4: {d['mean_pp']:.2f} pp "
                 f"[95% CI {d['ci95_low']:.2f}, {d['ci95_high']:.2f}],"
                 f"per-seed {d['per_seed']} → {'pass' if v['acc_pass'] else 'fail'}")
    lines.append(f"- GINE standalone ΔAcc vs E4: {dg['mean_pp']:.2f} pp "
                 f"[{dg['ci95_low']:.2f}, {dg['ci95_high']:.2f}]")
    lines.append(f"- vs E8 weighted_avg3 (anchor): {deltas['primary_vs_w3']['mean_pp']:+.2f} pp "
                 f"[{deltas['primary_vs_w3']['ci95_low']:.2f}, {deltas['primary_vs_w3']['ci95_high']:.2f}]")
    lines.append(f"- anchor recomputation weighted_avg3 = {v['anchor_weighted_avg3_mean']*100:.2f}% "
                 f"(E8 record 83.26%, |Δ|={v['anchor_abs_delta_pp']:.2f} pp) -> "
                 f"{'pass' if v['anchor_pass'] else 'fail'}\n")

    lines.append("## 2. 5-seed means per method (test)\n")
    lines.append("| Method | Acc | Macro-F1(sup) | Weighted-F1 | Top-5 |")
    lines.append("|---|---|---|---|---|")
    for m, a in agg.items():
        if m == "E4_XGBoost-104(full-train)":
            lines.append(f"| {m} | {a['acc_mean']*100:.2f} ± {a['acc_std']*100:.2f} | nan | nan | nan |")
        else:
            lines.append(f"| {m} | {a['acc_mean']*100:.2f} ± {a['acc_std']*100:.2f} | "
                         f"{a['macro_f1_support']['mean']*100:.2f} ± "
                         f"{a['macro_f1_support']['std']*100:.2f} | "
                         f"{a['weighted_f1']['mean']*100:.2f} ± {a['weighted_f1']['std']*100:.2f} | "
                         f"{a['top5']['mean']*100:.2f} ± {a['top5']['std']*100:.2f} |")
    lines.append("")
    lines.append("## 3. Guardrails (primary metric weighted_avg4)\n")
    lines.append(f"- Macro-F1(sup) mean {guard['macro_f1_support_mean']*100:.2f}% "
                 f"(ref {guard['macro_f1_support_ref']*100:.2f}%), per-seed pass "
                 f"{guard['macro_f1_support_seeds_pass'][0]}/{guard['macro_f1_support_seeds_pass'][1]}")
    lines.append(f"- Top-5 mean {guard['top5_mean']*100:.2f}% (ref {guard['top5_ref']*100:.2f}%),"
                 f"per-seed pass {guard['top5_seeds_pass'][0]}/{guard['top5_seeds_pass'][1]}\n")

    lines.append("## 4. Weight and complementarity diagnostics\n")
    lines.append(f"- GINE mean weight in weighted_avg4: {v['gine_weight_mean_pp']:.1f} pp,"
                 f"positive weights present: {v['gine_weight_positive']}")
    lines.append("- per-seed weights [XGB, BiGRU, FTT, GINE]: " +
                 "; ".join(f"seed {a['seed']}: {a['w4']}" for a in anchors))
    lines.append("- 4-model oracle (test set union hit rate): " +
                 ", ".join(f"seed {s}: {x*100:.2f}%" for s, x in v["oracle"].items()))
    for s, jm in v["error_jaccard"].items():
        lines.append(f"- seed {s} GINE vs three-model error Jaccard: " +
                     ", ".join(f"{m} {x:.3f}" for m, x in jm.items()))
    lines.append("")
    lines.append("## 5. Evidence boundary\n")
    lines.append("- GINE reads only the SMILES molecular graph, without the 104-dim tabular features such as elements/conditions/gel,"
                 "hence its information set differs from E8's three models; this is the core of the reopening condition and also why its standalone accuracy is limited. ")
    lines.append("- base models are all 5-fold in-fold models (80% training data per fold), paired with the E4 full-training-set XGBoost"
                 "There is a training-volume difference; this difference matches E8/E8b exactly and allows cross-experiment comparison. ")
    lines.append("- the 5-seed paired CI has only 4 degrees of freedom, so intervals are wide; the oracle only bounds complementary information, not realizable gains.")
    lines.append("")
    lines.append("## 6. Artifacts\n")
    lines.append("- `E8c_gnn_seed{42,123,777,2026,3407}.npz` / `E8c_graphs_cache.pt`")
    lines.append("- `E8c_seeds.csv` / `E8c_methods.csv` / `E8c_summary.json` / "
                 "`E8c_report.md` / `E8c_gnn_metrics.png` / `E8c_run.log`")
    with open(os.path.join(HERE, "E8c_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_metrics(agg, deltas, methods, seeds):
    neg = [m for m in methods if m != "E4_XGBoost-104(full-train)"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    means = [agg[m]["acc_mean"] * 100 for m in neg]
    stds = [agg[m]["acc_std"] * 100 for m in neg]
    axes[0].barh(neg, means, xerr=stds, color="#4C78A8")
    axes[0].axvline(agg["E4_XGBoost-104(full-train)"]["acc_mean"] * 100,
                    color="#D1495B", ls="--", label="E4 XGBoost")
    axes[0].set_xlabel("Test Accuracy (%)")
    axes[0].set_title("E8c method Acc (mean ± std, 5 seeds)")
    axes[0].legend()
    x = np.arange(len(seeds))
    w = 0.2
    for i, m in enumerate(neg):
        vals = [agg[m]["per_seed"][j] * 100 for j in range(len(seeds))]
        e4v = [agg["E4_XGBoost-104(full-train)"]["per_seed"][j] * 100 for j in range(len(seeds))]
        axes[1].bar(x + (i - 1.5) * w, np.array(vals) - np.array(e4v), width=w, label=m)
    axes[1].axhline(0.5, color="k", ls=":", label="+0.5 pp criterion")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(s) for s in seeds])
    axes[1].set_ylabel("ΔAcc vs E4 (pp)")
    axes[1].set_title("Per-seed paired differences")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "E8c_gnn_metrics.png"), dpi=160)
    plt.close(fig)


# ==================== main ====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()

    global LOG
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(os.path.join(HERE, "E8c_run.log"),
                                      mode="a" if args.analyze else "w",
                                      encoding="utf-8"),
                  logging.StreamHandler()])
    LOG = logging.getLogger("E8c")

    if args.analyze:
        analyze(args.seeds or SEEDS)
        return

    seeds = args.seeds or [args.seed]
    log("=" * 78)
    log(f"E8c new base model family | seeds={seeds} | smoke={args.smoke} | "
        f"device={ec.DEVICE} | torch={torch.__version__}")
    t0 = time.time()
    for s in seeds:
        run_seed(s, smoke=args.smoke, force=args.force)
    log(f"[main] all seeds done {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
