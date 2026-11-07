"""
E1.4 XGBoost-104 reference tuning (Optuna TPE + MedianPruner, 50 trials)
================================================================
Companion documents: CBUTransfer/DeepLearning/DeepLearning_Supplementary_Plan.md  §2.4 (E1.4)
Unified protocol: same as document §1 (data/split/metrics/seeds/fairness)

Purpose:
  satisfying fairness principle 1 -- when deep models (E1.1-E1.3) are tuned with 50 trials, the XGBoost-104 baseline
  also needs retuning under an equal budget, avoiding the asymmetry of "tuning deep models but not tree models".

Experiment design:
  1. Reproduce the baseline: Baseline104 default params (n_estimators=300, max_depth=10, lr=0.05, ...)
     evaluate on val/test for comparison with the tuned results.
  2. Optuna tuning: TPE(seed=42) + MedianPruner(n_startup_trials=10, warmup=50 trials)
     Search space (plan §2.4): n_estimators 100-1000 (log), max_depth 3-12,
     learning_rate 0.01-0.3(log), subsample/colsample_bytree 0.6-1.0,
     min_child_weight 1-20(log), reg_lambda 1e-3-10(log);
     gamma=0.1, reg_alpha=0.1 fixed as baseline values.
     Objective function: validation Macro-F1; fitting uses early_stopping_rounds=50.
  3. Retrain with best params and evaluate all metrics on test (Acc/Macro-F1/Weighted-F1/Top-5).

Outputs (all saved in this directory):
  - optuna_study_E1_4.db        Optuna study persistence (resumable/traceable)
  - best_params.json            best hyperparameters
  - trials_history.csv          full trial history (status, duration, each parameter)
  - E1_4_summary.json           baseline vs tuned comparison summary
  - E1_4_summary.csv            unified-schema result table of plan §12
  - E1_4_report.md              auto-generated experiment report (for traceability)
  - tuned_model.json            tuned model (xgboost native format)
  - e1_4_run.log                run log
"""

import os
import json
import time
import logging
import warnings

import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score

warnings.filterwarnings("ignore")

# ==================== Paths and constants ====================
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))  # Git/ (repo root holding data_processed/)
DATA_DIR = os.path.join(ROOT, "data_processed")
TRAIN_PATH = os.path.join(DATA_DIR, "ZEOSYN_104_struct_train.xlsx")
VAL_PATH = os.path.join(DATA_DIR, "ZEOSYN_104_struct_val.xlsx")
TEST_PATH = os.path.join(DATA_DIR, "ZEOSYN_104_struct_test.xlsx")

N_TRIALS = 50          # budget on par with deep models E1.1-E1.3 (plan §2.2)
SEED = 42              # unified protocol §1.3 default seed
EARLY_STOP_ROUNDS = 50
N_CLASS_EXPECT = 231

# Baseline104 default parameters (code1_prediction_104.py, reported test Acc=83.15%)
BASELINE_PARAMS = dict(
    n_estimators=300, max_depth=10, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9, min_child_weight=3,
    gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
)

# Plan section 2.4 search space
SEARCH_SPACE_DOC = {
    "n_estimators": "int 100-1000 (log)",
    "max_depth": "int 3-12",
    "learning_rate": "float 0.01-0.3 (log)",
    "subsample": "float 0.6-1.0",
    "colsample_bytree": "float 0.6-1.0",
    "min_child_weight": "float 1-20 (log)",
    "reg_lambda": "float 1e-3-10 (log)",
    "gamma": "Fixed 0.1 (baseline value)",
    "reg_alpha": "Fixed 0.1 (baseline value)",
}

# ==================== Logging (file + console dual write) ====================
log_path = os.path.join(HERE, "e1_4_run.log")
logger = logging.getLogger("E1_4")
logger.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
for h in (logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()):
    h.setFormatter(_fmt)
    logger.addHandler(h)


def log(msg):
    logger.info(msg)


# ==================== 104-feature definitions (consistent with Baseline104/bigru_104) ====================
FEATURES_104 = [
    "Si", "Al", "P", "Na", "K", "Li", "Sr", "Rb", "Cs", "Ba", "Ca", "F", "Ge",
    "Ti", "In", "B", "Mg", "Ga", "Ni", "Mn", "Fe", "Co", "Cr", "Zn", "Nb", "Be",
    "W", "Ce", "Cu", "Sn", "Gd", "La", "Y", "Dy", "Sm", "Ag", "Cd", "Zr", "V",
    "Ta", "Ru", "Hf", "Yb", "Tl", "As",
    "osda1_index", "osda2_index", "osda3_index",
    "cryst_temp", "cryst_time", "seed", "rotation",
    "aging_temp", "aging_time",
    "acid", "OH",
    "H2O_T", "OH_T", "Gel_Si_Al", "Gel_P_Al", "Gel_P_Si",
    "osda1_bertz_ct_mean_0", "osda1_free_sasa_mean_0", "osda1_asphericity_mean_0",
    "osda1_eccentricity_mean_0", "osda1_axes_mean_0", "osda1_axes_mean_1",
    "osda1_box_mean_0", "osda1_box_mean_1", "osda1_box_mean_2",
    "osda1_getaway_mean_0", "osda1_getaway_mean_1",
    "osda2_bertz_ct_mean_0", "osda2_free_sasa_mean_0", "osda2_asphericity_mean_0",
    "osda2_eccentricity_mean_0", "osda2_axes_mean_0", "osda2_axes_mean_1",
    "osda2_box_mean_0", "osda2_box_mean_1", "osda2_box_mean_2",
    "osda2_getaway_mean_0", "osda2_getaway_mean_1",
    "osda3_bertz_ct_mean_0", "osda3_free_sasa_mean_0", "osda3_asphericity_mean_0",
    "osda3_eccentricity_mean_0", "osda3_axes_mean_0", "osda3_axes_mean_1",
    "osda3_box_mean_0", "osda3_box_mean_1", "osda3_box_mean_2",
    "osda3_getaway_mean_0", "osda3_getaway_mean_1",
    "osda_avg_asphericity", "osda_max_asphericity", "osda_min_asphericity",
    "osda_avg_sasa", "osda_max_sasa", "osda_min_sasa",
    "osda_avg_bertz", "osda_max_bertz", "osda_min_bertz", "osda_total_volume",
]


# ==================== Data loading (protocol section 1.1, same as baseline: fillna(0)) ====================
def load_data():
    log("=" * 64)
    log("loading data (unified protocol §1.1)")
    train_df = pd.read_excel(TRAIN_PATH)
    val_df = pd.read_excel(VAL_PATH)
    test_df = pd.read_excel(TEST_PATH)
    log(f"train {train_df.shape} | val {val_df.shape} | test {test_df.shape}")

    feats = [f for f in FEATURES_104 if f in train_df.columns]
    if len(feats) != len(FEATURES_104):
        raise RuntimeError(f"features incomplete: missing {set(FEATURES_104) - set(feats)}")
    log(f"all 104 features available")

    le = LabelEncoder()
    X_train = train_df[feats].fillna(0)
    y_train = le.fit_transform(train_df["Code1_index"])
    X_val = val_df[feats].fillna(0)
    y_val = le.transform(val_df["Code1_index"])
    X_test = test_df[feats].fillna(0)
    y_test = le.transform(test_df["Code1_index"])
    log(f"classes: {len(le.classes_)} (expected {N_CLASS_EXPECT})")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), le


# ==================== Training and evaluation ====================
def make_model(params, trial=None):
    # xgboost>=2.0: callbacks must be passed to the constructor (fit() no longer accepts that parameter)
    callbacks = None
    if trial is not None:
        try:
            from optuna.integration import XGBoostPruningCallback
            callbacks = [XGBoostPruningCallback(trial, "validation_0-mlogloss")]
        except Exception as e:  # pragma: no cover
            log(f"pruning callback unavailable ({e}), no intermediate pruning for this trial")
    kwargs = dict(
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
        eval_metric="mlogloss",
        early_stopping_rounds=EARLY_STOP_ROUNDS,
        **params,
    )
    if callbacks is not None:
        kwargs["callbacks"] = callbacks
    return xgb.XGBClassifier(**kwargs), None


def evaluate(model, X, y, n_classes):
    # y is LabelEncoder integer encoded (0..n_classes-1); labels must use integer indices, not le.classes_ (strings)
    proba = model.predict_proba(X)
    pred = model.predict(X)
    return dict(
        acc=float(accuracy_score(y, pred)),
        macro_f1=float(f1_score(y, pred, average="macro")),
        weighted_f1=float(f1_score(y, pred, average="weighted")),
        top5=float(top_k_accuracy_score(y, proba, k=5, labels=np.arange(n_classes))),
        best_iteration=int(getattr(model, "best_iteration", -1)),
    )


def fit_eval(params, data, trial=None):
    (X_train, y_train), (X_val, y_val), (X_test, y_test), le = data
    model, _ = make_model(params, trial)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model, {
        "val": evaluate(model, X_val, y_val, len(le.classes_)),
        "test": evaluate(model, X_test, y_test, len(le.classes_)),
    }


# ==================== Optuna objective ====================
def objective(trial):
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 1000, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        min_child_weight=trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        gamma=0.1,
        reg_alpha=0.1,
    )
    _, scores = fit_eval(params, DATA, trial=trial)
    return scores["val"]["macro_f1"]


# ==================== Main pipeline ====================
def main():
    global DATA
    t0 = time.time()
    DATA = load_data()

    # ---- 1. Reproduce baseline (default parameters) ----
    log("-" * 64)
    log("[1/3] reproduce baseline: Baseline104 default parameters")
    base_model, base_scores = fit_eval(BASELINE_PARAMS, DATA)
    log(f"baseline  val Macro-F1={base_scores['val']['macro_f1']:.4f} | "
        f"test Acc={base_scores['test']['acc']:.4f} "
        f"Macro-F1={base_scores['test']['macro_f1']:.4f} "
        f"Weighted-F1={base_scores['test']['weighted_f1']:.4f} "
        f"Top-5={base_scores['test']['top5']:.4f} "
        f"(compared with the reported 83.15%)")

    # ---- 2. Optuna tuning ----
    log("-" * 64)
    log(f"[2/3] Optuna TPE tuning: {N_TRIALS} trials (seed={SEED})")
    storage = f"sqlite:///{os.path.join(HERE, 'optuna_study_E1_4.db')}"
    study = optuna.create_study(
        study_name="E1_4_xgboost104",
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=50),
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    done = len([t for t in study.trials if t.state.is_finished()])
    if done:
        log(f"found {done} completed trials, resuming to {N_TRIALS}")

    def _cb(study_, trial_):
        if trial_.state.is_finished():
            log(f"trial {trial_.number:>2d} | val MacroF1={trial_.value:.4f} "
                f"| best={study_.best_value:.4f} | {trial_.duration.total_seconds():.0f}s")

    remaining = N_TRIALS - done
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, callbacks=[_cb])

    # ---- 3. Retrain with best parameters + test evaluation ----
    log("-" * 64)
    log("[3/3] best-parameter retraining and final evaluation")
    best_params = dict(study.best_params)
    best_params.update(gamma=0.1, reg_alpha=0.1)
    log(f"best trial #{study.best_trial.number}: {json.dumps(best_params)}")
    tuned_model, tuned_scores = fit_eval(best_params, DATA)

    # Save the model (xgboost native format, with feature names, for traceability)
    tuned_model.save_model(os.path.join(HERE, "tuned_model.json"))

    # ---- 4. Summary output ----
    delta = {k: tuned_scores["test"][k] - base_scores["test"][k]
             for k in ("acc", "macro_f1", "weighted_f1", "top5")}

    with open(os.path.join(HERE, "best_params.json"), "w", encoding="utf-8") as f:
        json.dump({"best_trial": study.best_trial.number, "params": best_params,
                   "val_macro_f1": study.best_value}, f, indent=2, ensure_ascii=False)

    # trial history
    hist = study.trials_dataframe()
    hist.to_csv(os.path.join(HERE, "trials_history.csv"), index=False, encoding="utf-8-sig")

    # Plan section 12 unified schema
    rows = [
        dict(exp_id="E1.4", model="XGBoost-104", config_id="baseline_default",
             seed=SEED, fraction=1.0, n_params=np.nan,
             train_val_gap=round(base_scores["val"]["acc"] - base_scores["test"]["acc"], 4),
             **{"test_acc": round(base_scores["test"]["acc"], 4),
                "test_macro_f1": round(base_scores["test"]["macro_f1"], 4),
                "test_weighted_f1": round(base_scores["test"]["weighted_f1"], 4),
                "test_top5": round(base_scores["test"]["top5"], 4)},
             head_f1=np.nan, torso_f1=np.nan, tail_f1=np.nan,
             notes="Baseline104 default-parameter reproduction"),
        dict(exp_id="E1.4", model="XGBoost-104", config_id=f"tuned_trial{study.best_trial.number}",
             seed=SEED, fraction=1.0, n_params=np.nan,
             train_val_gap=round(tuned_scores["val"]["acc"] - tuned_scores["test"]["acc"], 4),
             **{"test_acc": round(tuned_scores["test"]["acc"], 4),
                "test_macro_f1": round(tuned_scores["test"]["macro_f1"], 4),
                "test_weighted_f1": round(tuned_scores["test"]["weighted_f1"], 4),
                "test_top5": round(tuned_scores["test"]["top5"], 4)},
             head_f1=np.nan, torso_f1=np.nan, tail_f1=np.nan,
             notes=f"Optuna TPE 50trials, val MacroF1={study.best_value:.4f}"),
    ]
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "E1_4_summary.csv"),
                              index=False, encoding="utf-8-sig")

    summary = dict(
        exp_id="E1.4", model="XGBoost-104", protocol="Supplementary Plan §1+§2.4",
        n_trials=N_TRIALS, finished_trials=len(
            [t for t in study.trials if t.state.is_finished()]),
        pruned_trials=len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        seed=SEED, search_space=SEARCH_SPACE_DOC,
        baseline=dict(params=BASELINE_PARAMS,
                      val=base_scores["val"], test=base_scores["test"],
                      reported_acc=0.8315),
        tuned=dict(trial_number=study.best_trial.number, params=best_params,
                   val=tuned_scores["val"], test=tuned_scores["test"]),
        delta_test=delta,
        wall_time_sec=round(time.time() - t0, 1),
    )
    with open(os.path.join(HERE, "E1_4_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    _write_report(summary)

    log(f"done, total time {summary['wall_time_sec']}s")
    log(f"conclusion: test Acc {base_scores['test']['acc']:.4f} -> {tuned_scores['test']['acc']:.4f} "
        f"(Δ={delta['acc']:+.4f}); Macro-F1 {base_scores['test']['macro_f1']:.4f} -> "
        f"{tuned_scores['test']['macro_f1']:.4f} (Δ={delta['macro_f1']:+.4f})")


def _write_report(s):
    b, t = s["baseline"], s["tuned"]
    md = f"""# E1.4 XGBoost-104 controlled tuning report

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')} | Companion: DeepLearning_Supplementary_Plan.md §2.4

## 1. Experiment purpose
satisfying principle 1 of the three fairness principles: deep models (E1.1-E1.3) each tuned with 50 trials, XGBoost-104 baseline
must be retuned under an equal budget, ruling out the asymmetric critiques of "insufficient deep model tuning" and "untuned tree models".

## 2. Experiment protocol
- Data: `data_processed/ZEOSYN_104_struct_train/val/test.xlsx` (12,537/2,640/2,641), frozen splits
- Features: 104 dims (all available); missing values fillna(0); labels Code1_index, LabelEncoder (231 classes)
- Tree models are scale-invariant, no StandardScaler applied (consistent with Baseline104; the deep models' standardization requirement does not apply)
- Seeds: {s['seed']}; early stopping: 50 rounds (monitoring val mlogloss); objective: val Macro-F1
- Note: early stopping and tuning share the validation set, as designed in plan §2.2; generalization conclusions are based on test

## 3. Search space (TPE, {s['n_trials']} trials, MedianPruner startup=10/warmup=50)
| Parameter | Range |
|------|------|
""" + "\n".join(f"| {k} | {v} |" for k, v in s["search_space"].items()) + f"""

completed {s['finished_trials']} trials, of which {s['pruned_trials']} pruned.

## 4. Result comparison

| Metric | baseline (default params) | tuned (trial #{t['trial_number']}) | Δ |
|------|---------------|----------------------------------|---|
| test Acc | {b['test']['acc']:.4f} (reported value {b['reported_acc']:.2%}) | {t['test']['acc']:.4f} | {s['delta_test']['acc']:+.4f} |
| test Macro-F1 | {b['test']['macro_f1']:.4f} | {t['test']['macro_f1']:.4f} | {s['delta_test']['macro_f1']:+.4f} |
| test Weighted-F1 | {b['test']['weighted_f1']:.4f} | {t['test']['weighted_f1']:.4f} | {s['delta_test']['weighted_f1']:+.4f} |
| test Top-5 | {b['test']['top5']:.4f} | {t['test']['top5']:.4f} | {s['delta_test']['top5']:+.4f} |
| val Acc | {b['val']['acc']:.4f} | {t['val']['acc']:.4f} | — |
| Early stopping rounds | {b['test'].get('best_iteration', -1)} | {t['test'].get('best_iteration', -1)} | — |

## 5. Best hyperparameters
```json
{json.dumps(t['params'], indent=2)}
```

## 6. Conclusions and write-back notes
1. Baseline changes after tuning: test Acc {s['delta_test']['acc']:+.2f} pp, Macro-F1 {s['delta_test']['macro_f1']:+.2f} pp.
2. If |Δ Acc| < 0.5 pp: report numbers unchanged, this report serves as evidence that "the default config is near the search-space ceiling".
   (original default parameter values kept in parentheses); the gap of deep models to it is recomputed.
3. If |Δ| < 0.5 pp: report numbers unchanged, this report serves as evidence that "the baseline is near the search-space ceiling".
4. E3 learning curve, E4 multi-seed and E8 stacking all use this experiment's tuned XGBoost configuration.
5. Deep models E1.1-E1.3 must use exactly the same budget ({s['n_trials']} trials) and protocol.

## 7. File list
| File | Content |
|------|------|
| optuna_study_E1_4.db | Optuna study (SQLite), can be reloaded with load_if_exists for retrospection |
| trials_history.csv | all trial parameters/status/elapsed time |
| best_params.json | best parameters and val score |
| E1_4_summary.json | machine-readable summary (baseline/tuning/Δ) |
| E1_4_summary.csv | plan §12 unified schema |
| tuned_model.json | tuned model (xgboost native format) |
| e1_4_run.log | run log |
"""
    with open(os.path.join(HERE, "E1_4_report.md"), "w", encoding="utf-8") as f:
        f.write(md)


if __name__ == "__main__":
    main()
