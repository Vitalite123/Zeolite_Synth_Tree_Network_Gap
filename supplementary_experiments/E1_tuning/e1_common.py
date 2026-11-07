"""
E1 common module: shared Optuna tuning skeleton for the three ANNs (E1.1 LeNet-5 / E1.2 BiGRU / E1.3 Transformer)
=====================================================================================
Companion documents: CBUTransfer/DeepLearning/DeepLearning_Supplementary_Plan.md §1 (unified protocol) + §2 (E1)
Reuse: E2_sota/e2_common.py's data loading (npz cache) / metrics / long-tail strata / random seeds

Differences from E2:
  - Training loop supports optimizer (Adam/AdamW) / scheduler (none/plateau/cosine+warmup) /
    label smoothing / gradient clipping, to faithfully replicate the training details of the three original ANNs;
  - Each trial reports val Macro-F1 per epoch and supports MedianPruner intermediate pruning;
  - Output prefixes differ per model (E1_1 / E1_2 / E1_3), and retrained model weights are saved as well.
"""

import gc
import json
import logging
import math
import os
import sys
import time

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(os.path.abspath(__file__))
E2_DIR = os.path.abspath(os.path.join(HERE, "..", "E2_sota"))
if E2_DIR not in sys.path:
    sys.path.insert(0, E2_DIR)
import e2_common as ec  # noqa: E402

DEVICE = ec.DEVICE
SEED = ec.SEED
N_TRIALS = 50

# All three models' forward is model(x) -> logits
FORWARD = lambda m, xb: m(xb)  # noqa: E731


def make_logger(name, filename):
    path = os.path.join(HERE, filename)
    lg = logging.getLogger(name)
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
        for h in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()):
            h.setFormatter(fmt)
            lg.addHandler(h)
    return lg


# ==================== Training loop ====================
def train_model(build_fn, params, data, logger, trial=None, max_epochs=100,
                patience=15, tag="", optimizer_kind="adamw", scheduler_kind="none",
                label_smoothing=0.0, grad_clip=None, warmup_ratio=None):
    """MX450 2GB: on CUDA OOM, automatically halve the batch and retry (up to 2 times, floor 64), consistent with E2."""
    bs = int(params["batch_size"])
    if DEVICE.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
    for attempt in range(3):
        try:
            return _train_once(build_fn, params, bs, data, logger, trial, max_epochs,
                               patience, tag, optimizer_kind, scheduler_kind,
                               label_smoothing, grad_clip, warmup_ratio, attempt)
        except torch.OutOfMemoryError:
            if attempt == 2 or bs // 2 < 64:
                raise
            bs //= 2
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            logger.info(f"{tag} CUDA OOM, halving batch to {bs} and retrying")


def _val_probs_loss(model, data, batch=256):
    """One forward pass obtains both val probabilities and mean cross-entropy (the plateau scheduler needs val loss)."""
    model.eval()
    outs, losses, n = [], 0.0, 0
    y_val = data["y_val"]
    with torch.no_grad():
        for i in range(0, len(data["X_val"]), batch):
            xb = torch.as_tensor(data["X_val"][i:i + batch], device=DEVICE)
            yb = torch.as_tensor(y_val[i:i + batch], dtype=torch.long, device=DEVICE)
            logits = FORWARD(model, xb)
            losses += float(F.cross_entropy(logits, yb, reduction="sum").item())
            n += len(yb)
            outs.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(outs), losses / max(n, 1)


def _train_once(build_fn, params, bs, data, logger, trial, max_epochs, patience, tag,
                optimizer_kind, scheduler_kind, label_smoothing, grad_clip,
                warmup_ratio, attempt):
    ec.set_seed(SEED)
    train_ds = TensorDataset(torch.as_tensor(data["X_train"]),
                             torch.as_tensor(data["y_train"], dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0,
                              pin_memory=(DEVICE.type == "cuda"))

    model = build_fn(params).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    Opt = torch.optim.AdamW if optimizer_kind == "adamw" else torch.optim.Adam
    lr = float(params.get("lr", params.get("learning_rate")))
    opt = Opt(model.parameters(), lr=lr,
              weight_decay=float(params.get("weight_decay", 0.0)))

    sched = None
    total_steps = max_epochs * max(1, len(train_loader))
    if scheduler_kind == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min",
                                                           factor=0.5, patience=5)
    elif scheduler_kind == "cosine_warmup":
        warm = max(1, int(float(warmup_ratio or 0.1) * total_steps))

        def lr_lambda(step):
            if step < warm:
                return step / warm
            prog = (step - warm) / max(1, total_steps - warm)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best_f1, best_state, best_epoch, bad = -1.0, None, 0, 0
    curve = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        total, n_batches = 0.0, 0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            loss = F.cross_entropy(FORWARD(model, xb), yb, label_smoothing=label_smoothing)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            opt.step()
            if scheduler_kind == "cosine_warmup":
                sched.step()
            total += float(loss.item())
            n_batches += 1

        probs, val_loss = _val_probs_loss(model, data)
        val_f1 = ec.metrics_from_proba(probs, data["y_val"], data["n_classes"])["macro_f1"]
        if scheduler_kind == "plateau":
            sched.step(val_loss)
        curve.append((epoch, total / max(n_batches, 1), val_loss, val_f1))

        if trial is not None:
            trial.report(val_f1, epoch)
            if trial.should_prune():
                del model
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()
                raise optuna.TrialPruned(f"epoch {epoch}: val MacroF1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1, best_epoch, bad = val_f1, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    metrics = {
        "val": ec.metrics_from_proba(
            ec.predict_proba(model, data["X_val"], FORWARD), data["y_val"],
            data["n_classes"], data.get("strata")),
        "test": ec.metrics_from_proba(
            ec.predict_proba(model, data["X_test"], FORWARD), data["y_test"],
            data["n_classes"], data.get("strata")),
    }
    metrics["n_params"] = n_params
    metrics["best_epoch"] = best_epoch
    metrics["train_val_gap"] = float(metrics["val"]["acc"] - metrics["test"]["acc"])
    logger.info(f"{tag} best_epoch={best_epoch}/{len(curve)} | val MacroF1={best_f1:.4f} "
                f"| test Acc={metrics['test']['acc']:.4f} "
                f"MacroF1={metrics['test']['macro_f1']:.4f} | params={n_params}")
    if DEVICE.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
    return model, metrics, curve, best_epoch


# ==================== Artifact export ====================
def _write_trials(study, prefix):
    rows = []
    for t in study.trials:
        row = {"number": t.number, "state": str(t.state).split(".")[-1],
               "val_macro_f1": t.value,
               "duration_s": round((t.datetime_complete - t.datetime_start).total_seconds(), 1)
               if t.datetime_complete and t.datetime_start else ""}
        row.update(t.params)
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(HERE, f"trials_history_{prefix}.csv"),
                              index=False, encoding="utf-8-sig")


def _write_summary(prefix, rows):
    cols = ec.SUMMARY_COLUMNS
    pd.DataFrame([{k: r.get(k, "") for k in cols} for r in rows], columns=cols).to_csv(
        os.path.join(HERE, f"{prefix}_summary.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(HERE, f"{prefix}_summary.json"), "w", encoding="utf-8") as f:
        json.dump([{k: r.get(k, "") for k in cols} for r in rows], f,
                  ensure_ascii=False, indent=2, default=str)


def _rows(prefix, model_name, ref_metrics, tuned_metrics, best_trial, best_val_f1):
    ref = ec.summary_row(prefix, model_name, "default_config", ref_metrics,
                         notes="original script default hyperparameters, rerun under the unified protocol")
    tuned = ec.summary_row(prefix, model_name, f"tuned_trial{best_trial}", tuned_metrics,
                           notes=f"Optuna TPE {N_TRIALS} trials, val MacroF1={best_val_f1:.4f}")
    return [ref, tuned]


def run_experiment(exp_id, model_name, prefix, build_fn, ref_params, suggest_fn,
                   train_kw_fn, ref_train_kw, max_epochs, patience,
                   search_space_doc, reported_acc, logger, t0):
    """Full pipeline: default baseline -> Optuna 50 trials -> retrain best config -> artifacts written to disk."""
    report_path = os.path.join(HERE, f"{prefix}_report.md")
    if os.path.exists(report_path):
        logger.info(f"{prefix} already complete (report exists: {report_path}), skipping; "
                    f"to fully rerun, delete this report and optuna_study_{prefix}.db")
        return

    data = ec.load_data(logger)

    # ---- 1. Default-config baseline (rerun under the unified protocol as the "before tuning" reference) ----
    logger.info("-" * 64)
    logger.info("[1/3] default config baseline (%s)" % json.dumps(ref_params, default=str))
    _, ref_metrics, _, ref_epoch = train_model(
        build_fn, ref_params, data, logger, trial=None, max_epochs=max_epochs,
        patience=patience, tag="ref", **ref_train_kw)
    logger.info(f"baseline test Acc={ref_metrics['test']['acc']:.4f} | value recorded in report={reported_acc:.4f}")

    # ---- 2. Optuna TPE tuning ----
    logger.info("-" * 64)
    logger.info(f"[2/3] Optuna TPE: {N_TRIALS} trials (seed={SEED})")
    storage = f"sqlite:///{os.path.join(HERE, f'optuna_study_{prefix}.db')}"
    study = optuna.create_study(
        study_name=f"{prefix}_{model_name}",
        storage=storage, load_if_exists=True, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10))
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    done = len([t for t in study.trials
                if t.state in (optuna.trial.TrialState.COMPLETE,
                               optuna.trial.TrialState.PRUNED)])

    def _kw(params):
        kw = dict(max_epochs=max_epochs, patience=patience)
        kw.update(train_kw_fn(params))
        return kw

    def objective(trial):
        params = suggest_fn(trial)
        _, m, _, _ = train_model(build_fn, params, data, logger, trial=trial,
                                 tag=f"trial{trial.number}", **_kw(params))
        return m["val"]["macro_f1"]

    def _cb(study_, trial_):
        if trial_.state.is_finished():
            logger.info(f"trial {trial_.number:>2d} | "
                        f"val MacroF1={trial_.value:.4f} | best={study_.best_value:.4f} | "
                        f"{trial_.duration.total_seconds():.0f}s")

    if done:
        logger.info(f"found {done} COMPLETE+PRUNED trials, resuming to {N_TRIALS}")
    remaining = N_TRIALS - done
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, callbacks=[_cb])

    # ---- 3. Retrain best configuration ----
    logger.info("-" * 64)
    logger.info("[3/3] best-config retraining and final evaluation")
    best_params = dict(study.best_params)
    logger.info(f"best trial #{study.best_trial.number}: {json.dumps(best_params)}")
    tuned_model, tuned_metrics, tuned_curve, tuned_epoch = train_model(
        build_fn, best_params, data, logger, trial=None, tag="tuned", **_kw(best_params))

    torch.save({"model_state_dict": tuned_model.state_dict(),
                "config": best_params, "exp_id": exp_id,
                "n_params": tuned_metrics["n_params"]},
               os.path.join(HERE, f"{prefix}_tuned.pt"))
    with open(os.path.join(HERE, f"best_params_{prefix}.json"), "w", encoding="utf-8") as f:
        json.dump({"exp_id": exp_id, "best_trial": study.best_trial.number,
                   "params": best_params, "val_macro_f1": study.best_value,
                   "n_params": tuned_metrics["n_params"]}, f,
                  indent=2, ensure_ascii=False)
    _write_trials(study, prefix)

    rows = _rows(prefix, model_name, ref_metrics, tuned_metrics,
                 study.best_trial.number, study.best_value)
    _write_summary(prefix, rows)

    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    delta = {k: tuned_metrics["test"][k] - ref_metrics["test"][k]
             for k in ("acc", "macro_f1", "weighted_f1", "top5")}

    summary = dict(
        exp_id=exp_id, model=model_name, protocol="Supplementary Plan §1+§2",
        n_trials=N_TRIALS, complete_trials=n_complete, pruned_trials=n_pruned,
        seed=SEED, max_epochs=max_epochs, patience=patience,
        search_space=search_space_doc,
        reference=dict(params=ref_params, best_epoch=ref_epoch,
                       val=ref_metrics["val"], test=ref_metrics["test"],
                       n_params=ref_metrics["n_params"]),
        tuned=dict(trial_number=study.best_trial.number, params=best_params,
                   best_epoch=tuned_epoch, val=tuned_metrics["val"],
                   test=tuned_metrics["test"], n_params=tuned_metrics["n_params"]),
        delta_test=delta, reported_acc=reported_acc,
        wall_time_sec=round(time.time() - t0, 1),
    )
    with open(os.path.join(HERE, f"{prefix}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    _write_report(prefix, summary)

    logger.info(f"done, total time {summary['wall_time_sec']}s")
    logger.info(f"conclusion: test Acc {ref_metrics['test']['acc']:.4f} -> "
                f"{tuned_metrics['test']['acc']:.4f} (Δ={delta['acc']:+.4f}); "
                f"Macro-F1 {ref_metrics['test']['macro_f1']:.4f} -> "
                f"{tuned_metrics['test']['macro_f1']:.4f} (Δ={delta['macro_f1']:+.4f})")


def _write_report(prefix, s):
    r, t = s["reference"], s["tuned"]
    delta = s["delta_test"]
    md = f"""# {s['exp_id']} {s['model']} Optuna tuning report

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')} | Companion: DeepLearning_Supplementary_Plan.md §2

## 1. Experiment purpose
answering the "deep models were not deeply hyperparameter-tuned" critique: under exactly the same budget as E1.4 (XGBoost-104)
({s['n_trials']} trials) and under the unified protocol, confirm the architectural ceiling of {s['model']},
give a "before tuning vs after tuning" comparison.

## 2. Experiment protocol
- Data: `data_processed/ZEOSYN_104_struct_train/val/test.xlsx` (12,537/2,640/2,641), frozen splits
- Features: 104 dims; missing values fillna(0); StandardScaler fitted on train only; labels Code1_index, 231 classes
- Seeds: {s['seed']}; early stopping: monitor val Macro-F1, patience={s['patience']}, max epochs={s['max_epochs']}
- Objective: val Macro-F1; sampler: TPE(seed={s['seed']}); pruner: MedianPruner(startup=10, warmup=10)
- Note: early stopping and tuning share the validation set (planned design in §2.2); generalization conclusions are based on test

## 3. Search space
| Parameter | Range |
|------|------|
""" + "\n".join(f"| {k} | {v} |" for k, v in s["search_space"].items()) + f"""

completed {s['complete_trials']} trials, pruned {s['pruned_trials']} (budget counted as COMPLETE+PRUNED).

## 4. Before/after tuning comparison

| Metric | before tuning (default hyperparameters) | after tuning (trial #{t['trial_number']}) | Δ |
|------|-----------------|--------------------------------|---|---|
| test Acc | {r['test']['acc']:.4f} | {t['test']['acc']:.4f} | {delta['acc']:+.4f} |
| test Macro-F1 | {r['test']['macro_f1']:.4f} | {t['test']['macro_f1']:.4f} | {delta['macro_f1']:+.4f} |
| test Weighted-F1 | {r['test']['weighted_f1']:.4f} | {t['test']['weighted_f1']:.4f} | {delta['weighted_f1']:+.4f} |
| test Top-3 | {r['test']['top3']:.4f} | {t['test']['top3']:.4f} | — |
| test Top-5 | {r['test']['top5']:.4f} | {t['test']['top5']:.4f} | {delta['top5']:+.4f} |
| test Top-10 | {r['test']['top10']:.4f} | {t['test']['top10']:.4f} | — |
| head / torso / tail F1 | {r['test']['head_f1']:.4f} / {r['test']['torso_f1']:.4f} / {r['test']['tail_f1']:.4f} | {t['test']['head_f1']:.4f} / {t['test']['torso_f1']:.4f} / {t['test']['tail_f1']:.4f} | — |
| Param count | {r['n_params']:,} | {t['n_params']:,} | — |
| Best epoch | {r['best_epoch']} | {t['best_epoch']} | — |

report's originally recorded value (test Acc): {s['reported_acc']:.4f}

## 5. Best hyperparameters
```json
{json.dumps(t['params'], indent=2)}
```

## 6. Conclusions and write-back rules
1. After tuning, test Acc changes {delta['acc'] * 100:+.2f} pp and Macro-F1 changes {delta['macro_f1'] * 100:+.2f} pp.
2. If |Δ Acc| >= 0.5 pp: update {s['model']}'s numbers in full report §4.1 and the abstract to tuned values (originals kept in parentheses).
3. If |Δ Acc| < 0.5 pp: numbers unchanged, this report serves as evidence that "the default config is near the search-space ceiling".
4. This experiment's tuned configuration is reused by E3 (learning curve) / E4 (multi-seed) / E8 (stacking).

## 7. File list
| File | Content |
|------|------|
| optuna_study_{prefix}.db | Optuna study (SQLite), supports resume |
| trials_history_{prefix}.csv | all trial parameters/status/elapsed time |
| best_params_{prefix}.json | best parameters and val score |
| {prefix}_summary.json | machine-readable summary (baseline/tuning/Δ) |
| {prefix}_summary.csv | plan unified schema (19 columns) |
| {prefix}_tuned.pt | retrained model weights |
| {prefix.lower()}_run.log | run log |
"""
    with open(os.path.join(HERE, f"{prefix}_report.md"), "w", encoding="utf-8") as f:
        f.write(md)
