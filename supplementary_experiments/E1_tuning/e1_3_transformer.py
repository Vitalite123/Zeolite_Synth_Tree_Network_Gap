"""
E1.3 Transformer hyperparameter tuning (Optuna TPE + MedianPruner, 50 trials)
=====================================================================================
Companion documents: CBUTransfer/DeepLearning/DeepLearning_Supplementary_Plan.md §2.3 (E1.3)
Search space: plan §2.3 table; architecture reused from ../Transformer/transformer_104_fast.py
"""

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
E1_DIR = os.path.abspath(os.path.join(HERE))
if E1_DIR not in sys.path:
    sys.path.insert(0, E1_DIR)
import e1_common  # noqa: E402

import optuna
import torch
import torch.nn as nn
import numpy as np


# ==================== Model definitions ==== (identical to the original transformer_104_fast.py) ====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=20, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Transformer_Zeolite(nn.Module):
    def __init__(self, num_classes, d_model, nhead, n_layers,
                 dim_feedforward_mult, dropout):
        super().__init__()
        self.d_model = d_model
        self.seq_len = 13
        self.token_dim = 8

        self.feature_embedding = nn.Linear(self.token_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.seq_len + 1, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * dim_feedforward_mult,
            dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)
        if x.size(1) < self.seq_len * self.token_dim:
            padding = torch.zeros(batch_size, self.seq_len * self.token_dim - x.size(1),
                                  device=x.device)
            x = torch.cat([x, padding], dim=1)
        elif x.size(1) > self.seq_len * self.token_dim:
            x = x[:, :self.seq_len * self.token_dim]
        x = x.view(batch_size, self.seq_len, self.token_dim)
        x = self.feature_embedding(x)
        x = x * np.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x


# ==================== Search space and parameter extraction ====================
SEARCH_SPACE = {
    "d_model": "categorical {64, 128, 192, 256}",
    "nhead": "categorical {2, 4, 8} (d_model must be divisible)",
    "num_layers": "categorical {2, 3, 4, 6}",
    "dim_feedforward_mult": "categorical {2, 4}",
    "learning_rate": "log 5e-5 - 1e-3",
    "warmup_ratio": "categorical {0.05, 0.1}",
    "dropout_rate": "uniform 0.1 - 0.5",
    "batch_size": "categorical {64, 128, 256}",
}

REPORTED_ACC = 0.6778  # value recorded in report §4.1


def suggest_params(trial: optuna.Trial):
    dm = trial.suggest_categorical("d_model", [64, 128, 192, 256])
    # 64/128/192/256 are all divisible by 8, so the nhead candidate set can be unified statically
    # (SQLite-persisted studies require identical parameter distributions across all trials; cannot condition on d_model)
    nh = trial.suggest_categorical("nhead", [2, 4, 8])
    nl = trial.suggest_categorical("num_layers", [2, 3, 4, 6])
    dfm = trial.suggest_categorical("dim_feedforward_mult", [2, 4])
    lr = trial.suggest_float("learning_rate", 5e-5, 1e-3, log=True)
    wr = trial.suggest_categorical("warmup_ratio", [0.05, 0.1])
    dr = trial.suggest_float("dropout_rate", 0.1, 0.5)
    bs = trial.suggest_categorical("batch_size", [64, 128, 256])
    return dict(
        d_model=dm, nhead=nh, num_layers=nl, dim_feedforward_mult=dfm,
        learning_rate=lr, warmup_ratio=wr, dropout_rate=dr, batch_size=bs)


def build_model(params, n_classes):
    return Transformer_Zeolite(
        num_classes=n_classes,
        d_model=params["d_model"],
        nhead=params["nhead"],
        n_layers=params["num_layers"],
        dim_feedforward_mult=params["dim_feedforward_mult"],
        dropout=params["dropout_rate"])


def get_train_kw(params):
    return dict(
        optimizer_kind="adamw",
        label_smoothing=0.0, grad_clip=1.0,
        scheduler_kind="cosine_warmup",
        warmup_ratio=params["warmup_ratio"])


def get_ref_train_kw():
    # The original transformer_104_fast.py uses ReduceLROnPlateau (no warmup); the baseline replicates it
    return dict(optimizer_kind="adamw", label_smoothing=0.0, grad_clip=1.0,
                scheduler_kind="plateau")


def get_ref_params():
    return dict(
        d_model=64, nhead=4, num_layers=2, dim_feedforward_mult=2,
        learning_rate=0.001, warmup_ratio=0.05, dropout_rate=0.3, batch_size=128)


# ==================== Main entry ====================
def main():
    from e1_common import make_logger, run_experiment
    log_path = os.path.join(HERE, "e1_3_run.log")
    logger = make_logger("E1.3", log_path)
    t0 = time.time()
    run_experiment(
        exp_id="E1.3", model_name="Transformer", prefix="E1_3",
        build_fn=lambda p: build_model(p, 231),
        ref_params=get_ref_params(),
        suggest_fn=suggest_params,
        train_kw_fn=get_train_kw,
        ref_train_kw=get_ref_train_kw(),
        max_epochs=100,
        patience=15,
        search_space_doc=SEARCH_SPACE,
        reported_acc=REPORTED_ACC,
        logger=logger,
        t0=t0)


if __name__ == "__main__":
    main()
