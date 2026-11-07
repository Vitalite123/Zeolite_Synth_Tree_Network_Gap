# Data-Quality Adjudication of the Zeolite Framework Prediction Gap

Code and machine-readable artifacts for the study of the tree–network accuracy gap
on zeolite framework prediction from hydrothermal synthesis recipes
(104-feature pipeline, frozen stratified split, five-seed statistical protocol).

## Data provenance

The original dataset is **ZeoSyn**, published by:

> Elton Pan, Soonhyoung Kwon, Zach Jensen, Mingrou Xie, Rafael Gómez-Bombarelli,
> Manuel Moliner, Yuriy Román-Leshkov, Elsa Olivetti; *ZeoSyn: A Comprehensive
> Zeolite Synthesis Dataset Enabling Machine-Learning Rationalization of
> Hydrothermal Parameters*. **ACS Cent. Sci.** 2024, 10 (3), 729–743.

The cleaned 104-feature matrices, the frozen stratified split and the label mapping
in `data_processed/` are derived from ZeoSyn. Please cite the reference above when
using this repository.

## Repository layout

```
Git/
├── data_processed/             frozen split (train/val/test, 104 features) + code1_index_mapping.csv
├── supplementary_experiments/  E1–E9, E11: scripts + machine-readable outputs
│   ├── E1_tuning/              LeNet-5 / BiGRU / Transformer / XGBoost 50-trial Optuna tuning
│   ├── E2_sota/                FT-Transformer / TabNet / MLP-ResNet references
│   ├── E3_learning_curve/      learning-curve fits (acc = a + b·ln N)
│   ├── E3b_quality/            label-quality stratification
│   ├── E4_seeds/               five-seed replication + paired statistics (+ per-seed predictions)
│   ├── E5_imbalance/           class-imbalance controls
│   ├── E6_tokenization/        serialization-granularity and permutation controls
│   ├── E7_capacity/            capacity scans (+ per-seed predictions)
│   ├── E8_stacking/            stacking / GINE fusion study (registered negative result)
│   ├── E9_interpretability/    permutation, TreeSHAP / GradientSHAP, attention, masks
│   └── E11_dedup/              deduplicated-split rerun (+ per-seed predictions)
└── tables/                     main_results.csv, statistical_tests.csv, tuning_comparison.csv, ...
```

Each experiment folder contains its training/analysis script, the Optuna study
database, trial histories, best parameters, summaries (JSON/CSV), the retrained
model weights, and per-seed test-set prediction files where applicable.

## Environment

Python 3.14.7 with PyTorch 2.13.0 (CUDA 13.0 build), XGBoost 3.4.1, LightGBM 4.7.0,
scikit-learn 1.9.1, rtdl 0.0.13, pytorch-tabnet 4.1.0, PyTorch Geometric 2.8.0.post1,
RDKit 2026.03.6, Optuna 5.0.0 and shap 0.54.1. Training ran on an NVIDIA GeForce
RTX 5060 Laptop GPU (8 GB).

Scripts resolve data through paths relative to this repository root
(`data_processed/`); no absolute paths are used. The derived array cache
`ZEOSYN_104_struct_cache.npz` is regenerated automatically from the released
xlsx files on first use.

## License

Released under the MIT License (see `LICENSE`). The underlying ZeoSyn dataset
follows the terms of its original publication.
