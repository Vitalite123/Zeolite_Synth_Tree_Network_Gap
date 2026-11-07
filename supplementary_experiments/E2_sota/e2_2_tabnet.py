"""
E2.2 TabNet (pytorch-tabnet) 50-trial tuning
================================================================
Companion documents: CBUTransfer/DeepLearning/DeepLearning_Supplementary_Plan.md §3.2 (E2.2)
Protocol: aligned with E1_4 -- data/split/seed 42/objective val Macro-F1/metrics Acc|MacroF1|WeightedF1|Top-5
Model: pytorch-tabnet 4.1.0 TabNetClassifier (Arik & Pfister 2021)

Notes:
  - TabNet's training loop is library-managed: early stopping monitors eval_metric='logloss' (same as E1.4 XGBoost
    consistent with the mlogloss early-stopping protocol), the Optuna objective remains post-fit val Macro-F1
  - The library loop is invisible to Optuna -> no per-epoch intermediate pruning; trial duration is determined by its own early stopping
  - mask_type fixed to sparsemax (entmax requires extra interactive dependencies, not introduced)

Outputs (this directory): optuna_study_E2_2.db / best_params.json / trials_history.csv /
  E2_2_summary.{json,csv} / E2_2_report.md / e2_2_tabnet_tuned.zip (library-native format) /
  e2_2_tuned_training_curve.csv / e2_2_run.log
"""

import json
import os
import time
import gc

import numpy as np
import optuna
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
import pytorch_tabnet.metrics as _pt_metrics
from sklearn.metrics import log_loss as _sk_log_loss

from e2_common import (HERE, DEVICE, SEED, MAX_EPOCHS, PATIENCE,
                       make_logger, set_seed, load_data, metrics_from_proba,
                       summary_row, write_summary, write_trials_history, write_report)

EXP_ID = "E2.2"
PREFIX = "E2_2"
N_TRIALS = 50
LOGGER = make_logger("E2_2", "e2_2_run.log")
log = LOGGER.info

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Compatibility patch: pytorch-tabnet 4.1.0's built-in LogLoss calls sklearn log_loss without labels,
# When the validation set lacks full class coverage (<231 classes), the newer sklearn raises; also that version's check_metrics
# The custom Metric class degrades to a name string; custom metrics effectively break (get_metrics_by_names
# resolve by the same-named built-in class first). So we directly patch the built-in LogLoss.__call__, with labels as the only difference.
# ---------------------------------------------------------------------------
def _patched_logloss_call(self, y_true, y_score):
    return float(_sk_log_loss(y_true, y_score,
                              labels=np.arange(np.asarray(y_score).shape[1])))

_pt_metrics.LogLoss.__call__ = _patched_logloss_call


# ==================== Training and evaluation ====================
def fit_eval_tabnet(params, data, tag):
    """params: contains tabnet hyperparameters plus lr/weight_decay/batch_size"""
    set_seed(SEED)
    bs = params["batch_size"]
    if DEVICE.type == "cuda":  # empty the VRAM pool to reduce OOM/driver crash probability from fragmentation
        gc.collect()
        torch.cuda.empty_cache()
    clf = TabNetClassifier(
        n_d=params["n_d"], n_a=params["n_a"], n_steps=params["n_steps"],
        gamma=params["gamma"], n_independent=params["n_independent"],
        n_shared=params["n_shared"], momentum=params["momentum"],
        lambda_sparse=params["lambda_sparse"], mask_type="sparsemax",
        optimizer_params=dict(lr=params["lr"],
                              weight_decay=params["weight_decay"]),
        seed=SEED, verbose=0, device_name=str(DEVICE),
    )
    clf.fit(
        data["X_train"], data["y_train"],
        eval_set=[(data["X_val"], data["y_val"])],
        eval_metric=["logloss"],
        max_epochs=MAX_EPOCHS, patience=PATIENCE,
        batch_size=bs, virtual_batch_size=bs // 2,
        num_workers=0, drop_last=False,
    )
    n_classes = data["n_classes"]
    metrics = {
        "val": metrics_from_proba(clf.predict_proba(data["X_val"]),
                                  data["y_val"], n_classes, data.get("strata")),
        "test": metrics_from_proba(clf.predict_proba(data["X_test"]),
                                   data["y_test"], n_classes, data.get("strata")),
    }
    metrics["n_params"] = sum(int(v.numel()) for v in clf.network.parameters())
    # Early stopping uses val logloss (minimized); best_epoch = argmin+1 (history key: val_0_logloss)
    vll = list(clf.history["val_0_logloss"])
    metrics["best_epoch"] = int(np.argmin(vll)) + 1
    log(f"{tag} epochs={len(clf.history['loss'])} | "
        f"val MacroF1={metrics['val']['macro_f1']:.4f} | "
        f"test Acc={metrics['test']['acc']:.4f} "
        f"MacroF1={metrics['test']['macro_f1']:.4f} | params={metrics['n_params']}")
    return clf, metrics


def make_objective(data):
    def objective(trial):
        n_independent = trial.suggest_int("n_independent", 1, 3)
        params = dict(
            n_d=trial.suggest_int("n_d", 16, 64, step=8),
            n_a=trial.suggest_int("n_a", 16, 64, step=8),
            n_steps=trial.suggest_int("n_steps", 3, 6),
            gamma=trial.suggest_float("gamma", 1.2, 2.0),
            n_independent=n_independent,
            n_shared=min(trial.suggest_int("n_shared", 0, 2), n_independent),
            momentum=trial.suggest_float("momentum", 0.005, 0.1, log=True),
            lambda_sparse=trial.suggest_float("lambda_sparse", 1e-4, 1e-1, log=True),
            lr=trial.suggest_float("lr", 1e-3, 3e-2, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True),
            batch_size=trial.suggest_categorical("batch_size", [256, 512]),
        )
        clf, metrics = fit_eval_tabnet(params, data, f"trial {trial.number}")
        del clf
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        return metrics["val"]["macro_f1"]
    return objective


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
    log(f"[E2.2] TabNet tuning | device={DEVICE} | pytorch-tabnet 4.1.0")
    data = load_data(LOGGER)

    # ---- 1. Library default parameters reference ----
    log("-" * 64)
    log("[1/3] TabNet library default parameter reference (n_d=n_a=8, n_steps=3, lr=2e-2, batch=512)")
    default_params = dict(n_d=8, n_a=8, n_steps=3, gamma=1.3, n_independent=2,
                          n_shared=1, momentum=0.02, lambda_sparse=1e-3,
                          lr=2e-2, weight_decay=0.0, batch_size=512)
    # Reference-config result cache: skip duplicate training on crash restart
    ref_path = os.path.join(HERE, f"{PREFIX}_ref_default.json")
    if os.path.exists(ref_path):
        with open(ref_path, encoding="utf-8") as f:
            def_metrics = json.load(f)
        log(f"reusing cached reference config results: {PREFIX}_ref_default.json "
            f"(val MacroF1={def_metrics['val']['macro_f1']:.4f})")
    else:
        def_clf, def_metrics = fit_eval_tabnet(default_params, data, "tabnet_default")
        del def_clf
        if DEVICE.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
        with open(ref_path, "w", encoding="utf-8") as f:
            json.dump(def_metrics, f, ensure_ascii=False, indent=2, default=str)
    rows = [summary_row(EXP_ID, "TabNet-104", "tabnet_default", def_metrics,
                        notes="pytorch-tabnet library default hyperparameters (batch 512)")]

    # ---- 2. Optuna tuning ----
    log("-" * 64)
    log(f"[2/3] Optuna TPE x {N_TRIALS} trials (objective: val Macro-F1, early stopping logloss)")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10),
        study_name="E2_2_TabNet",
        storage=f"sqlite:///{os.path.join(HERE, 'optuna_study_E2_2.db')}",
        load_if_exists=True,
    )
    # Resumable: only count COMPLETE/PRUNED toward the budget; RUNNING/FAIL are crash leftovers and are not counted
    _valid = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    n_done = len([t for t in study.trials if t.state in _valid])
    if n_done >= N_TRIALS:
        log(f"resume: study already has {n_done}/{N_TRIALS} trials, skipping tuning")
    else:
        log(f"resume: study already has {n_done}/{N_TRIALS} trials, running {N_TRIALS - n_done} more")
        study.optimize(make_objective(data), n_trials=N_TRIALS - n_done,
                       gc_after_trial=True, show_progress_bar=False)
    write_trials_history(study, PREFIX)
    best = study.best_trial
    log(f"best trial #{best.number}: val MacroF1={best.value:.4f} | {json.dumps(best.params)}")
    with open(os.path.join(HERE, "best_params.json"), "w", encoding="utf-8") as f:
        json.dump({"trial": best.number, "val_macro_f1": best.value,
                   "params": best.params}, f, ensure_ascii=False, indent=2)

    # ---- 3. Retrain with best parameters + persist ----
    log("-" * 64)
    log(f"[3/3] best-parameter retraining (trial #{best.number})")
    params = dict(best.params)
    # The native crash root cause is pinned to the pandas/pyarrow string backend (README item 11), unrelated to VRAM;
    # Retraining reuses the tuned batch so the final numbers strictly match the trial config
    log(f"retraining batch_size={params['batch_size']} (keeping tuned config)")
    clf, metrics = fit_eval_tabnet(params, data, f"tuned_trial{best.number}")
    rows.append(summary_row(EXP_ID, "TabNet-104", f"tuned_trial{best.number}",
                            metrics,
                            notes=f"Optuna TPE {N_TRIALS}trials, "
                                  f"val MacroF1={best.value:.4f}"))
    write_summary(PREFIX, rows)
    saved_path = clf.save_model(os.path.join(HERE, "e2_2_tabnet_tuned"))
    log(f"model saved: {saved_path}")
    import pandas as pd
    pd.DataFrame({"epoch": range(1, len(clf.history["loss"]) + 1),
                  "train_loss": clf.history["loss"],
                  "val_logloss": clf.history["val_0_logloss"]}).to_csv(
        os.path.join(HERE, "e2_2_tuned_training_curve.csv"), index=False)

    write_report(os.path.join(HERE, f"{PREFIX}_report.md"),
                 f"{EXP_ID} TabNet-104 tuning report", [
        ("1. Experiment purpose", "satisfies fairness principle 2: complete the tabular deep-learning SOTA (TabNet) and the GBDT baseline"
                       "and the horizontal comparison of in-house ANNs. "),
        ("2. Experiment protocol",
         f"- Data: data_processed/ZEOSYN_104_struct_{{train,val,test}}.xlsx "
         f"(frozen splits), 104-dim features, fillna(0), StandardScaler (train fit only)\n"
         f"- Labels: Code1_index -> LabelEncoder ({data['n_classes']} classes)\n"
         f"- Device: {DEVICE} (torch {torch.__version__}, pytorch-tabnet 4.1.0)\n"
         f"- Seed: {SEED}; early stopping: val logloss patience {PATIENCE}, "
         f"max_epochs {MAX_EPOCHS} (consistent with the E1.4 XGBoost mlogloss early-stopping protocol);\n"
         f"  Optuna objective: val Macro-F1 after fit\n"
         f"- Optuna: TPE(seed={SEED}), {N_TRIALS} trials; "
         f"library-managed loop has no per-epoch intermediate pruning\n"
         f"- mask_type: sparsemax (fixed; entmax requires an extra dependency and is not introduced)"),
        ("3. Search space",
         "| Parameter | Range |\n|------|------|\n"
         "| n_d / n_a | int 16-64 (step 8, sampled independently) |\n| n_steps | int 3-6 |\n"
         "| gamma | 1.2-2.0 |\n| n_independent | int 1-3 |\n"
         "| n_shared | int 0-2 (truncated to <= n_independent) |\n"
         "| momentum | 0.005-0.1 (log) |\n| lambda_sparse | 1e-4 - 1e-1 (log) |\n"
         "| lr (Adam) | 1e-3 - 3e-2 (log) |\n| weight_decay | 1e-6 - 1e-4 (log) |\n"
         "| batch_size | {256, 512} (virtual = batch/2) |"),
        ("4. Result comparison",
         "| Config | test Acc | test Macro-F1 | Weighted-F1 | Top-5 | epochs |\n"
         "|------|---------|--------------|-------------|-------|--------|\n"
         f"| tabnet_default | {def_metrics['test']['acc']:.4f} | "
         f"{def_metrics['test']['macro_f1']:.4f} | "
         f"{def_metrics['test']['weighted_f1']:.4f} | "
         f"{def_metrics['test']['top5']:.4f} | {def_metrics['best_epoch']} |\n"
         f"| tuned_trial{best.number} | {metrics['test']['acc']:.4f} | "
         f"{metrics['test']['macro_f1']:.4f} | "
         f"{metrics['test']['weighted_f1']:.4f} | "
         f"{metrics['test']['top5']:.4f} | {metrics['best_epoch']} |"),
        ("5. Best hyperparameters",
         "```json\n" + json.dumps(best.params, ensure_ascii=False, indent=2) + "\n```"),
        ("6. Comparison anchors",
         "E1.4 XGBoost-104: baseline test Acc=0.8315 / Macro-F1=0.7397; "
         "tuned Acc=0.8353 / Macro-F1=0.7247 (plan section 2.4) "),
        ("7. File list",
         "| File | Content |\n|------|------|\n"
         "| optuna_study_E2_2.db | Optuna study (SQLite) |\n"
         "| best_params.json | best parameters |\n| trials_history.csv | full trial history |\n"
         "| E2_2_summary.json/csv | plan section 12 unified schema |\n"
         "| e2_2_tabnet_tuned.zip | tuned model (library-native format) |\n"
         "| e2_2_tuned_training_curve.csv | retraining curve |\n"
         "| e2_2_run.log | run log | "),
    ])
    log(f"done, total time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
