"""
E2.1 FT-Transformer (rtdl) 50-trial tuning
================================================================
Companion documents: CBUTransfer/DeepLearning/DeepLearning/DeepLearning_Supplementary_Plan.md §3.1 (E2.1)
Protocol: aligned with E1_4 -- data/split/seed 42/objective val Macro-F1/metrics Acc|MacroF1|WeightedF1|Top-5
Model: rtdl 0.0.13 FTTransformer (tabular SOTA reference, Gorishniy et al. 2021)

Notes:
  - rtdl pins torch<2 in all version metadata; installed with --no-deps; empirical code is compatible with torch 2.11
  - Prefer make_baseline (full search space); fall back to make_default (only n_blocks tunable) if unavailable
  - Early stopping: val Macro-F1 patience 15 (protocol §1.4); Optuna per-epoch report + MedianPruner

Outputs (this directory): optuna_study_E2_1.db / best_params.json / trials_history.csv /
  E2_1_summary.{json,csv} / E2_1_report.md / e2_1_ftt_tuned.pt /
  e2_1_tuned_training_curve.csv / e2_1_run.log
"""

import inspect
import json
import os
import time

import numpy as np
import optuna
import rtdl
import torch

from e2_common import (HERE, DEVICE, SEED, N_CLASS_EXPECT, MAX_EPOCHS, PATIENCE,
                       make_logger, set_seed, load_data, train_rtdl,
                       summary_row, write_summary, write_trials_history, write_report)

EXP_ID = "E2.1"
PREFIX = "E2_1"
N_TRIALS = 50
LOGGER = make_logger("E2_1", "e2_1_run.log")
log = LOGGER.info

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ==================== Model building (rtdl 0.0.13 API adaptive) ====================
def _baseline_sig():
    if not hasattr(rtdl.FTTransformer, "make_baseline"):
        return None
    return set(inspect.signature(rtdl.FTTransformer.make_baseline).parameters)


BASELINE_SIG = _baseline_sig()


def build_ftt(params, n_classes):
    """Built against the actual rtdl 0.0.13 signature; prenormalization=True (rtdl recommended)"""
    if BASELINE_SIG:
        kwargs = dict(
            n_num_features=104, cat_cardinalities=None, d_out=n_classes,
            d_token=params["d_token"], n_blocks=params["n_blocks"],
            attention_n_heads=params["n_heads"],
            ffn_d_hidden=int(params["d_token"] * params["ffn_mult"]),
            ffn_dropout=params["ffn_dropout"],
            residual_dropout=params["resid_dropout"],
            prenormalization=True,
        )
        for k in ("attention_dropout", "attention_probs_dropout"):
            if k in BASELINE_SIG:
                kwargs[k] = params["attn_dropout"]
                break
        kwargs = {k: v for k, v in kwargs.items() if k in BASELINE_SIG}
        return rtdl.FTTransformer.make_baseline(**kwargs)
    return rtdl.FTTransformer.make_default(
        n_num_features=104, cat_cardinalities=None, d_out=n_classes,
        n_blocks=params["n_blocks"])


def forward_ftt(model, xb):
    return model(xb, None)  # FTTransformer two-argument forward


# ==================== Optuna objective ====================
def make_objective(data):
    def objective(trial):
        d_token = trial.suggest_categorical("d_token", [32, 64, 96, 128])
        n_blocks = trial.suggest_int("n_blocks", 2, 5)
        # The batch candidate set must be static: SQLite storage requires identical parameter distributions across all trials
        # (previously the candidate set was conditioned on d_token/n_blocks; trial8 triggered a dynamic value space crash);
        # When MX450 VRAM is insufficient, train_rtdl's OOM auto-halving retry is the fallback
        params = dict(
            d_token=d_token,
            n_blocks=n_blocks,
            n_heads=trial.suggest_categorical("n_heads", [4, 8]),
            attn_dropout=trial.suggest_float("attn_dropout", 0.1, 0.3),
            ffn_mult=trial.suggest_categorical("ffn_mult", [4 / 3, 8 / 3, 4.0]),
            ffn_dropout=trial.suggest_float("ffn_dropout", 0.05, 0.2),
            resid_dropout=trial.suggest_float("resid_dropout", 0.0, 0.1),
            lr=trial.suggest_float("lr", 3e-5, 1e-3, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            batch_size=trial.suggest_categorical("batch_size", [256, 512]),
        )
        _, metrics, _, _ = train_rtdl(
            lambda p: build_ftt(p, data["n_classes"]), forward_ftt, params, data,
            LOGGER, trial=trial,
            tag=f"trial {trial.number}")
        return metrics["val"]["macro_f1"]
    return objective


# ==================== Main pipeline ====================
def main():
    t0 = time.time()
    # Resumable (idempotent): the final report is the last persisted artifact; if it exists, this script has fully run, so skip
    # E2_RETRAIN_ONLY=1: rerun only the "retrain best config" segment (tuning and reference configs use cache), to fix the batch convention
    _retrain_only = os.environ.get("E2_RETRAIN_ONLY") == "1"
    if os.path.exists(os.path.join(HERE, f"{PREFIX}_report.md")) and not _retrain_only:
        log("=" * 64)
        log(f"[{EXP_ID}] {PREFIX}_report.md exists, skipping entire script (resume)")
        return
    log("=" * 64)
    log(f"[E2.1] FT-Transformer tuning | device={DEVICE} | rtdl API: "
        f"{'make_baseline(full space)' if BASELINE_SIG else 'make_default(restricted space)'}")
    data = load_data(LOGGER)
    set_seed(SEED)

    # ---- 1. rtdl default-architecture reference (reference row, analogous to E1.4's baseline reproduction) ----
    log("-" * 64)
    log("[1/3] rtdl make_default architecture reference (d_token=64, n_blocks=3, lr=1e-4, wd=1e-5, batch=256)")
    default_params = dict(d_token=64, n_blocks=3, n_heads=8, attn_dropout=0.2,
                          ffn_mult=8 / 3, ffn_dropout=0.1, resid_dropout=0.0,
                          lr=1e-4, weight_decay=1e-5, batch_size=256)
    # Reference-config result cache: on crash restart, skips ~16 minutes of duplicate training (this row is reference only, bound to SEED)
    ref_path = os.path.join(HERE, f"{PREFIX}_ref_default.json")
    if os.path.exists(ref_path):
        with open(ref_path, encoding="utf-8") as f:
            def_metrics = json.load(f)
        log(f"reusing cached reference config results: {PREFIX}_ref_default.json "
            f"(val MacroF1={def_metrics['val']['macro_f1']:.4f})")
    else:
        _, def_metrics, _, _ = train_rtdl(
            lambda p: build_ftt(p, data["n_classes"]), forward_ftt, default_params, data,
            LOGGER, tag="rtdl_default")
        with open(ref_path, "w", encoding="utf-8") as f:
            json.dump(def_metrics, f, ensure_ascii=False, indent=2, default=str)
    rows = [summary_row(EXP_ID, "FT-Transformer-104", "rtdl_default", def_metrics,
                        notes="rtdl make_default architecture, recommended hyperparameters")]

    # ---- 2. Optuna tuning (TPE seed=42 + MedianPruner) ----
    log("-" * 64)
    log(f"[2/3] Optuna TPE x {N_TRIALS} trials (objective: val Macro-F1)")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10),
        study_name="E2_1_FTT",
        storage=f"sqlite:///{os.path.join(HERE, 'optuna_study_E2_1.db')}",
        load_if_exists=True,
    )
    # Resumable: only count COMPLETE/PRUNED toward the budget (both count as "evaluated" trials);
    # RUNNING and FAIL left over from crashes/pauses are not counted, otherwise the effective budget < 50, asymmetric with E1.4
    _valid = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    n_done = len([t for t in study.trials if t.state in _valid])
    _stale = len([t for t in study.trials
                  if t.state in (optuna.trial.TrialState.RUNNING,
                                 optuna.trial.TrialState.FAIL)])
    if _stale:
        log(f"ignoring {_stale} stale trials (RUNNING/FAIL, not counted against the tuning budget)")
    if n_done >= N_TRIALS:
        log(f"resume: study already has {n_done}/{N_TRIALS} trials, skipping tuning")
    else:
        log(f"resume: study already has {n_done}/{N_TRIALS} trials, running {N_TRIALS - n_done} more")
        study.optimize(make_objective(data), n_trials=N_TRIALS - n_done,
                       gc_after_trial=True, show_progress_bar=False)
    log(f"exporting trial history ({len(study.trials)} rows) ...")
    write_trials_history(study, PREFIX)
    log("trial history export complete")
    best = study.best_trial
    log(f"best trial #{best.number}: val MacroF1={best.value:.4f} | {json.dumps(best.params)}")
    with open(os.path.join(HERE, "best_params.json"), "w", encoding="utf-8") as f:
        json.dump({"trial": best.number, "val_macro_f1": best.value,
                   "params": best.params}, f, ensure_ascii=False, indent=2)

    # ---- 3. Retrain with best parameters + persist ----
    log("-" * 64)
    log(f"[3/3] best-parameter retraining (trial #{best.number})")
    params = dict(best.params)
    # The native crash root cause is pinned to the pandas/pyarrow string backend (see README item 11), unrelated to VRAM;
    # Hence retraining reuses the tuned batch so the final numbers strictly match the trial config
    log(f"retraining batch_size={params['batch_size']} (keeping tuned config)")
    model, metrics, curve, best_epoch = train_rtdl(
        lambda p: build_ftt(p, data["n_classes"]), forward_ftt, params, data,
        LOGGER, tag=f"tuned_trial{best.number}")
    rows.append(summary_row(EXP_ID, "FT-Transformer-104", f"tuned_trial{best.number}",
                            metrics, notes=f"Optuna TPE {N_TRIALS}trials, "
                                           f"val MacroF1={best.value:.4f}"))
    write_summary(PREFIX, rows)
    torch.save({"state_dict": model.state_dict(), "params": params,
                "best_epoch": best_epoch},
               os.path.join(HERE, "e2_1_ftt_tuned.pt"))
    import pandas as pd
    pd.DataFrame(curve, columns=["epoch", "train_loss", "val_macro_f1"]).to_csv(
        os.path.join(HERE, "e2_1_tuned_training_curve.csv"), index=False)

    write_report(os.path.join(HERE, f"{PREFIX}_report.md"),
                 f"{EXP_ID} FT-Transformer-104 tuning report", [
        ("1. Experiment purpose", "satisfies fairness principle 2: complete the tabular deep-learning SOTA (FT-Transformer) and"
                       "horizontal comparison of GBDT baselines and in-house ANNs (LeNet-5/BiGRU/Transformer). "),
        ("2. Experiment protocol", f"- Data: data_processed/ZEOSYN_104_struct_{{train,val,test}}.xlsx "
                       f"(frozen splits), 104-dim features, fillna(0), StandardScaler (train fit only)\n"
                       f"- Labels: Code1_index -> LabelEncoder ({data['n_classes']} classes)\n"
                       f"- Device: {DEVICE} (torch {torch.__version__}, rtdl 0.0.13)\n"
                       f"- Seed: {SEED}; early stopping: val Macro-F1 patience {PATIENCE}, "
                       f"max_epochs {MAX_EPOCHS}; objective: val Macro-F1\n"
                       f"- Optuna: TPE(seed={SEED}) + MedianPruner(startup=10, warmup=10), "
                       f"{N_TRIALS} trials"),
        ("3. Search space (make_baseline)" if BASELINE_SIG else "3. Search space (make_default, restricted)",
         "| Parameter | Range |\n|------|------|\n"
         "| d_token | {32, 64, 96, 128} |\n| n_blocks | int 2-5 |\n"
         "| attention_n_heads | {4, 8} |\n| attention_dropout | 0.1-0.3 |\n"
         "| ffn_d_hidden | d_token x {4/3, 8/3, 4} |\n| ffn_dropout | 0.05-0.2 |\n"
         "| residual_dropout | 0.0-0.1 |\n| lr (AdamW) | 3e-5 - 1e-3 (log) |\n"
         "| weight_decay | 1e-6 - 1e-3 (log) |\n| batch_size | {256, 512} |\n"
         "| prenormalization | True (fixed, rtdl recommended) | "),
        ("4. Result comparison",
         "| Config | test Acc | test Macro-F1 | Weighted-F1 | Top-5 | best_epoch |\n"
         "|------|---------|--------------|-------------|-------|------------|\n"
         f"| rtdl_default | {def_metrics['test']['acc']:.4f} | "
         f"{def_metrics['test']['macro_f1']:.4f} | "
         f"{def_metrics['test']['weighted_f1']:.4f} | "
         f"{def_metrics['test']['top5']:.4f} | {def_metrics['best_epoch']} |\n"
         f"| tuned_trial{best.number} | {metrics['test']['acc']:.4f} | "
         f"{metrics['test']['macro_f1']:.4f} | "
         f"{metrics['test']['weighted_f1']:.4f} | "
         f"{metrics['test']['top5']:.4f} | {best_epoch} |"),
        ("5. Best hyperparameters", "```json\n" + json.dumps(best.params, ensure_ascii=False, indent=2)
         + "\n```"),
        ("6. Comparison anchors",
         "E1.4 XGBoost-104: baseline test Acc=0.8315 / Macro-F1=0.7397; "
         "tuned Acc=0.8353 / Macro-F1=0.7247 (plan section 2.4) "),
        ("7. File list",
         "| File | Content |\n|------|------|\n"
         "| optuna_study_E2_1.db | Optuna study (SQLite) |\n"
         "| best_params.json | best parameters |\n| trials_history.csv | full trial history |\n"
         "| E2_1_summary.json/csv | plan section 12 unified schema |\n"
         "| e2_1_ftt_tuned.pt | tuned model weights |\n"
         "| e2_1_tuned_training_curve.csv | retraining curve |\n"
         "| e2_1_run.log | run log | "),
    ])
    log(f"done, total time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
