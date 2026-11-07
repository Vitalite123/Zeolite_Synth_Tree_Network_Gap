"""
E2.3 MLP / ResNet (rtdl) reference models, 25+25-trial tuning
================================================================
Companion documents: CBUTransfer/DeepLearning/DeepLearning_Supplementary_Plan.md §3.3 (E2.3)
Protocol: aligned with E1_4 -- data/split/seed 42/objective val Macro-F1/metrics Acc|MacroF1|WeightedF1|Top-5
Model: rtdl 0.0.13 MLP and ResNet (classic references for tabular deep learning, Gorishniy et al. 2021)

Notes:
  - Reference models get half the budget (25 trials each, plan §3.3); FT-Transformer (E2.1) / TabNet (E2.2) get 50 each
  - MLP/ResNet use the single-input forward(x); FT-Transformer uses forward(x_num, x_cat)
  - Early stopping: val Macro-F1 patience 15; Optuna per-epoch report + MedianPruner

Outputs (this directory): optuna_study_E2_3_{MLP,ResNet}.db / best_params.json /
  trials_history.csv / E2_3_summary.{json,csv} / E2_3_report.md /
  e2_3_{mlp,resnet}_tuned.pt / e2_3_{mlp,resnet}_tuned_curve.csv / e2_3_run.log
"""

import json
import os
import time
import gc

import numpy as np
import optuna
import pandas as pd
import rtdl
import torch

from e2_common import (HERE, DEVICE, SEED, MAX_EPOCHS, PATIENCE,
                       make_logger, set_seed, load_data, train_rtdl,
                       summary_row, write_summary, write_report)

EXP_ID = "E2.3"
PREFIX = "E2_3"
N_TRIALS = 25  # reference models get half the budget (plan §3.3)
LOGGER = make_logger("E2_3", "e2_3_run.log")
log = LOGGER.info

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ==================== Model building (rtdl 0.0.13: single-argument forward) ====================
def build_mlp(params, n_classes):
    return rtdl.MLP.make_baseline(
        d_in=104, d_layers=[params["width"]] * params["n_layers"],
        dropout=params["dropout"], d_out=n_classes)


def build_resnet(params, n_classes):
    return rtdl.ResNet.make_baseline(
        d_in=104, n_blocks=params["n_blocks"], d_main=params["d_main"],
        d_hidden=params["d_main"] * 2, dropout_first=params["dropout_first"],
        dropout_second=params["dropout_second"], d_out=n_classes)


def forward_single(model, xb):
    return model(xb)  # MLP/ResNet single-argument forward


# ==================== Optuna objective ====================
def make_objective_mlp(data):
    def objective(trial):
        params = dict(
            n_layers=trial.suggest_int("n_layers", 1, 3),
            width=trial.suggest_categorical("width", [128, 256, 512]),
            dropout=trial.suggest_float("dropout", 0.1, 0.5),
            lr=trial.suggest_float("lr", 1e-4, 1e-3, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            batch_size=trial.suggest_categorical("batch_size", [256, 512]),
        )
        _, metrics, _, _ = train_rtdl(
            lambda p: build_mlp(p, data["n_classes"]), forward_single, params, data,
            LOGGER, trial=trial, tag=f"MLP trial {trial.number}")
        return metrics["val"]["macro_f1"]
    return objective


def make_objective_resnet(data):
    def objective(trial):
        d_main = trial.suggest_categorical("d_main", [128, 256])
        params = dict(
            n_blocks=trial.suggest_int("n_blocks", 1, 4),
            d_main=d_main,
            dropout_first=trial.suggest_float("dropout_first", 0.0, 0.3),
            dropout_second=trial.suggest_float("dropout_second", 0.0, 0.3),
            lr=trial.suggest_float("lr", 1e-4, 1e-3, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            batch_size=trial.suggest_categorical("batch_size", [256, 512]),
        )
        _, metrics, _, _ = train_rtdl(
            lambda p: build_resnet(p, data["n_classes"]), forward_single, params, data,
            LOGGER, trial=trial, tag=f"ResNet trial {trial.number}")
        return metrics["val"]["macro_f1"]
    return objective


# ==================== Single-model pipeline ====================
def run_model(data, name, build_fn, ref_params, objective, rows):
    # Reference configs (result cache: skip duplicate training when restarting after a crash)
    log("-" * 64)
    log(f"[{name}] rtdl recommended reference configuration")
    ref_path = os.path.join(HERE, f"{PREFIX}_{name.lower()}_ref.json")
    if os.path.exists(ref_path):
        with open(ref_path, encoding="utf-8") as f:
            ref_metrics = json.load(f)
        log(f"[{name}] reusing cached reference config results: {os.path.basename(ref_path)}")
    else:
        _, ref_metrics, _, _ = train_rtdl(
            lambda p: build_fn(p, data["n_classes"]), forward_single, ref_params, data,
            LOGGER, tag=f"{name}_ref")
        with open(ref_path, "w", encoding="utf-8") as f:
            json.dump(ref_metrics, f, ensure_ascii=False, indent=2, default=str)
    rows.append(summary_row(EXP_ID, f"{name}-104", "rtdl_ref", ref_metrics,
                            notes="rtdl recommended reference configuration"))

    # Tuning
    log(f"[{name}] Optuna TPE x {N_TRIALS} trials")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10),
        study_name=f"E2_3_{name}",
        storage=f"sqlite:///{os.path.join(HERE, f'optuna_study_E2_3_{name}.db')}",
        load_if_exists=True,
    )
    # Resumable: only count COMPLETE/PRUNED toward the budget; RUNNING/FAIL are crash leftovers and are not counted
    _valid = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    n_done = len([t for t in study.trials if t.state in _valid])
    if n_done >= N_TRIALS:
        log(f"[{name}] resume: study already has {n_done}/{N_TRIALS} trials, skipping tuning")
    else:
        log(f"[{name}] resume: study already has {n_done}/{N_TRIALS} trials, "
            f"run {N_TRIALS - n_done} remaining trials")
        study.optimize(objective, n_trials=N_TRIALS - n_done, gc_after_trial=True,
                       show_progress_bar=False)
    best = study.best_trial
    log(f"[{name}] best trial #{best.number}: val MacroF1={best.value:.4f} | "
        f"{json.dumps(best.params)}")

    # Retraining (the pandas/pyarrow native-crash root cause is fixed; reuse the tuned batch)
    _rt = dict(best.params)
    log(f"[{name}] retraining batch_size={_rt['batch_size']} (keeping tuned config)")
    model, metrics, curve, best_epoch = train_rtdl(
        lambda p: build_fn(p, data["n_classes"]), forward_single, _rt,
        data, LOGGER, tag=f"{name}_tuned_trial{best.number}")
    rows.append(summary_row(EXP_ID, f"{name}-104", f"tuned_trial{best.number}",
                            metrics,
                            notes=f"Optuna TPE {N_TRIALS}trials, "
                                  f"val MacroF1={best.value:.4f}"))
    torch.save({"state_dict": model.state_dict(), "params": dict(best.params),
                "best_epoch": best_epoch},
               os.path.join(HERE, f"e2_3_{name.lower()}_tuned.pt"))
    pd.DataFrame(curve, columns=["epoch", "train_loss", "val_macro_f1"]).to_csv(
        os.path.join(HERE, f"e2_3_{name.lower()}_tuned_curve.csv"), index=False)
    return name, best, metrics, ref_metrics, study


# ==================== Main pipeline ====================
def main():
    t0 = time.time()
    # Resumable (idempotent): the final report is the last persisted artifact; if it exists, this script has fully run, so skip
    # E2_RETRAIN_ONLY=1: rerun only the "retrain best config" segment, to fix the batch convention
    _retrain_only = os.environ.get("E2_RETRAIN_ONLY") == "1"
    if os.path.exists(os.path.join(HERE, f"{PREFIX}_report.md")) and not _retrain_only:
        log("=" * 64)
        log(f"[{EXP_ID}] {PREFIX}_report.md exists, skipping entire script (resume)")
        return
    log("=" * 64)
    log(f"[E2.3] MLP / ResNet reference model tuning | device={DEVICE}")
    data = load_data(LOGGER)
    set_seed(SEED)
    rows = []

    mlp_ref = dict(n_layers=2, width=256, dropout=0.2, lr=1e-4,
                   weight_decay=1e-5, batch_size=256)
    resnet_ref = dict(n_blocks=2, d_main=128, dropout_first=0.2, dropout_second=0.0,
                      lr=1e-4, weight_decay=1e-5, batch_size=256)

    mlp_res = run_model(data, "MLP", build_mlp, mlp_ref,
                        make_objective_mlp(data), rows)
    resnet_res = run_model(data, "ResNet", build_resnet, resnet_ref,
                           make_objective_resnet(data), rows)

    # ---- Merge trial histories ----
    frames = []
    for name, _, _, _, study in (mlp_res, resnet_res):
        df = pd.DataFrame([{"study": name, "number": t.number,
                            "state": str(t.state).split(".")[-1], "value": t.value}
                           | t.params for t in study.trials])
        frames.append(df)
    pd.concat(frames, ignore_index=True).to_csv(
        os.path.join(HERE, f"{PREFIX}_trials_history.csv"), index=False,
        encoding="utf-8")

    # ---- best_params.json (both models) ----
    with open(os.path.join(HERE, "best_params.json"), "w", encoding="utf-8") as f:
        json.dump({name: {"trial": best.number, "val_macro_f1": best.value,
                          "params": best.params}
                   for name, best, _, _, _ in (mlp_res, resnet_res)},
                  f, ensure_ascii=False, indent=2)
    write_summary(PREFIX, rows)

    def _row(m):
        return (f"| {m['test']['acc']:.4f} | {m['test']['macro_f1']:.4f} | "
                f"{m['test']['weighted_f1']:.4f} | {m['test']['top5']:.4f} |")

    (name_m, best_m, m_m, ref_m, _), (name_r, best_r, m_r, ref_r, _) = mlp_res, resnet_res
    write_report(os.path.join(HERE, f"{PREFIX}_report.md"),
                 f"{EXP_ID} MLP/ResNet-104 reference model tuning report", [
        ("1. Experiment purpose", "add the classic tabular deep-learning references (MLP/ResNet), together with FT-Transformer (E2.1),"
                       "TabNet (E2.2), in-house ANNs, and GBDT baselines form a full gradient; 25 trials each. "),
        ("2. Experiment protocol",
         f"- Data: data_processed/ZEOSYN_104_struct_{{train,val,test}}.xlsx "
         f"(frozen splits), 104-dim features, fillna(0), StandardScaler (train fit only)\n"
         f"- Labels: Code1_index -> LabelEncoder ({data['n_classes']} classes)\n"
         f"- Device: {DEVICE} (torch {torch.__version__}, rtdl 0.0.13)\n"
         f"- Seed: {SEED}; early stopping: val Macro-F1 patience {PATIENCE}, "
         f"max_epochs {MAX_EPOCHS}; objective: val Macro-F1\n"
         f"- Optuna: TPE(seed={SEED}) + MedianPruner(startup=10, warmup=10), "
         f"MLP {N_TRIALS} + ResNet {N_TRIALS} trials"),
        ("3. Search space",
         "**MLP**: n_layers 1-3 x width {128,256,512} (d_layers=[width]*n_layers), "
         "dropout 0.1-0.5, lr 1e-4-1e-3(log), wd 1e-6-1e-3(log), batch {256,512}\n\n"
         "**ResNet**: n_blocks 1-4, d_main {128,256}, d_hidden=2*d_main, "
         "dropout_first/second 0.0-0.3, lr/wd/batch as above"),
        ("4. Result comparison",
         "| Model | Config | test Acc | test Macro-F1 | Weighted-F1 | Top-5 |\n"
         "|------|------|---------|--------------|-------------|-------|\n"
         f"| MLP | rtdl_ref |{_row(ref_m)} |\n"
         f"| MLP | tuned_trial{best_m.number} |{_row(m_m)} |\n"
         f"| ResNet | rtdl_ref |{_row(ref_r)} |\n"
         f"| ResNet | tuned_trial{best_r.number} |{_row(m_r)} |"),
        ("5. Best hyperparameters",
         "**MLP**\n```json\n" + json.dumps(best_m.params, ensure_ascii=False, indent=2)
         + "\n```\n**ResNet**\n```json\n"
         + json.dumps(best_r.params, ensure_ascii=False, indent=2) + "\n```"),
        ("6. Comparison anchors",
         "E1.4 XGBoost-104: baseline test Acc=0.8315 / Macro-F1=0.7397; "
         "tuned Acc=0.8353 / Macro-F1=0.7247 (plan section 2.4) "),
        ("7. File list",
         "| File | Content |\n|------|------|\n"
         "| optuna_study_E2_3_MLP.db / _ResNet.db | Optuna study |\n"
         "| best_params.json | best parameters of both models |\n"
         "| E2_3_trials_history.csv | merged trial history (with study column) |\n"
         "| E2_3_summary.json/csv | plan section 12 unified schema |\n"
         "| e2_3_mlp_tuned.pt / e2_3_resnet_tuned.pt | tuned weights |\n"
         "| e2_3_{mlp,resnet}_tuned_curve.csv | retraining curves |\n"
         "| e2_3_run.log | run log | "),
    ])
    log(f"done, total time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
