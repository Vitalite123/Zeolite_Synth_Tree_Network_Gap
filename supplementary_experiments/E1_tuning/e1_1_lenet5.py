"""
E1.1 LeNet-5 hyperparameter tuning (Optuna TPE + MedianPruner, 50 trials)
=====================================================================================
Companion documents: CBUTransfer/DeepLearning/DeepLearning_Supplementary_Plan.md §2.3 (E1.1)
Search space: plan §2.3 table; architecture reused from LeNet5Model.py / lenet5_104_quick.py
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


# ==================== Model definitions ==== (identical to the original lenet5_104_quick.py) ====================
class LeNet5_Zeolite(nn.Module):
    def __init__(self, num_classes, conv_channels=(32, 64, 128), dropout_rate=0.3):
        super().__init__()
        c1, c2, c3 = conv_channels

        self.conv1 = nn.Conv1d(1, c1, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(c1, c2, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(c2, c3, kernel_size=3, stride=1, padding=1)

        conv_output_size = 26 * c3  # 104 -> pool 52 -> pool 26; conv3 padding=1 keeps 26
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(256, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        return x


# ==================== Search space and parameter extraction ====================
SEARCH_SPACE = {
    "learning_rate": "log 1e-4 - 3e-3",
    "dropout_rate": "uniform 0.2 - 0.6",
    "batch_size": "categorical {64, 128, 256}",
    "conv_channels": "categorical {(32,64,128), (64,128,256), (64,64,128)}",
    "weight_decay": "log 1e-5 - 1e-3",
}

REPORTED_ACC = 0.7736  # value recorded in report §4.1


def suggest_params(trial: optuna.Trial):
    lr = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
    dropout = trial.suggest_float("dropout_rate", 0.2, 0.6)
    batch = trial.suggest_categorical("batch_size", [64, 128, 256])
    # Use strings instead of tuples: Optuna's SQLite storage deserializes tuples to lists, breaking resume value lookup
    conv = trial.suggest_categorical("conv_channels", ["32-64-128",
                                                      "64-128-256",
                                                      "64-64-128"])
    wd = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    return dict(
        learning_rate=lr, dropout_rate=dropout, batch_size=batch,
        conv_channels=tuple(int(c) for c in conv.split("-")), weight_decay=wd)


def build_model(params, n_classes):
    # study.best_params returns the string keys persisted at save time ("64-128-256"); parse them back into channel tuples
    cc = params["conv_channels"]
    if isinstance(cc, str):
        cc = tuple(int(c) for c in cc.split("-"))
    return LeNet5_Zeolite(
        num_classes=n_classes,
        conv_channels=cc,
        dropout_rate=params["dropout_rate"])


def get_train_kw(params):
    return dict(
        optimizer_kind="adam",
        label_smoothing=0.0, grad_clip=1.0,
        scheduler_kind="none")


def get_ref_params():
    return dict(
        learning_rate=0.001, dropout_rate=0.3, batch_size=128,
        conv_channels=(64, 128, 256), weight_decay=0.0)


# ==================== Main entry ====================
def main():
    from e1_common import make_logger, run_experiment
    log_path = os.path.join(HERE, "e1_1_run.log")
    logger = make_logger("E1.1", log_path)
    t0 = time.time()
    run_experiment(
        exp_id="E1.1", model_name="LeNet-5", prefix="E1_1",
        build_fn=lambda p: build_model(p, 231),
        ref_params=get_ref_params(),
        suggest_fn=suggest_params,
        train_kw_fn=get_train_kw,
        ref_train_kw=get_train_kw(get_ref_params()),
        max_epochs=60,
        patience=10,
        search_space_doc=SEARCH_SPACE,
        reported_acc=REPORTED_ACC,
        logger=logger,
        t0=t0)


if __name__ == "__main__":
    main()
