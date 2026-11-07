"""
E1.2 BiGRU hyperparameter tuning (Optuna TPE + MedianPruner, 50 trials)
=====================================================================================
Companion documents: CBUTransfer/DeepLearning/DeepLearning_Supplementary_Plan.md §2.3 (E1.2)
Search space: plan §2.3 table; architecture reused from ../BiGRU/bigru_104.py
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


# ==================== Model definitions ==== (identical to the original bigru_104.py) ====================
class BiGRU_Zeolite(nn.Module):
    def __init__(self, num_classes, hidden_size, num_layers, dropout, attention_dim):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Reshape 104 features to (seq_len=26, input_size=4)
        self.seq_len = 26
        self.feature_per_step = 4

        self.feature_embedding = nn.Linear(self.feature_per_step, hidden_size // 2)

        self.gru = nn.GRU(
            input_size=hidden_size // 2, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0)

        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        self.fc1 = nn.Linear(hidden_size * 2, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.seq_len, self.feature_per_step)
        x = self.feature_embedding(x)
        gru_out, _ = self.gru(x)
        attn_weights = self.attention(gru_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * gru_out, dim=1)

        out = self.fc1(context)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        return out


# ==================== Search space and parameter extraction ====================
SEARCH_SPACE = {
    "hidden_size": "categorical {64, 128, 256, 512}",
    "num_layers": "categorical {1, 2, 3}",
    "learning_rate": "log 1e-4 - 3e-3",
    "dropout_rate": "uniform 0.2 - 0.6",
    "batch_size": "categorical {64, 128, 256}",
    "attention_dim": "categorical {64, 128, 256}",
    "weight_decay": "log 1e-5 - 1e-3",
}

REPORTED_ACC = 0.8016  # value recorded in report §4.1


def suggest_params(trial: optuna.Trial):
    hs = trial.suggest_categorical("hidden_size", [64, 128, 256, 512])
    nl = trial.suggest_categorical("num_layers", [1, 2, 3])
    lr = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
    dropout = trial.suggest_float("dropout_rate", 0.2, 0.6)
    batch = trial.suggest_categorical("batch_size", [64, 128, 256])
    ad = trial.suggest_categorical("attention_dim", [64, 128, 256])
    wd = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    return dict(
        hidden_size=hs, num_layers=nl, learning_rate=lr,
        dropout_rate=dropout, batch_size=batch, attention_dim=ad, weight_decay=wd)


def build_model(params, n_classes):
    return BiGRU_Zeolite(
        num_classes=n_classes,
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        dropout=params["dropout_rate"],
        attention_dim=params["attention_dim"])


def get_train_kw(params):
    return dict(
        optimizer_kind="adamw",
        label_smoothing=0.1, grad_clip=1.0,
        scheduler_kind="plateau")


def get_ref_params():
    return dict(
        hidden_size=128, num_layers=2, learning_rate=0.001,
        dropout_rate=0.3, batch_size=128, attention_dim=128, weight_decay=0.01)


# ==================== Main entry ====================
def main():
    from e1_common import make_logger, run_experiment
    log_path = os.path.join(HERE, "e1_2_run.log")
    logger = make_logger("E1.2", log_path)
    t0 = time.time()
    run_experiment(
        exp_id="E1.2", model_name="BiGRU", prefix="E1_2",
        build_fn=lambda p: build_model(p, 231),
        ref_params=get_ref_params(),
        suggest_fn=suggest_params,
        train_kw_fn=get_train_kw,
        ref_train_kw=get_train_kw(get_ref_params()),
        max_epochs=100,
        patience=15,
        search_space_doc=SEARCH_SPACE,
        reported_acc=REPORTED_ACC,
        logger=logger,
        t0=t0)


if __name__ == "__main__":
    main()
