# TAPT: Task-Identity as Sufficient Inductive Bias for Molecular Property Prediction

This repository contains the official implementation of **TAPT** (Task-Adaptive Prompt Tuning), a parameter-efficient prompt tuning framework for molecular property prediction.

---

## Overview

TAPT is a minimalist prompt tuning framework built on pre-trained Communicative Message Passing Neural Networks (CMPNN). Unlike prior knowledge-augmented approaches that inject external chemical knowledge graphs into prompts (e.g., KANO), **TAPT operates without any external ontology or chemical knowledge base**. It relies solely on a lightweight, learnable **task identity embedding** as the inductive bias for downstream task adaptation.

![Architecture Overview](./fig/overview.png)

### Core Mechanism

TAPT introduces a learnable **task identity embedding** combined with a lightweight cross-attention module that produces a task-conditioned molecular prompt:

1. **Pre-trained Encoding** — A CMPNN encoder pre-trained on ZINC15 via self-supervised contrastive objectives produces molecule-level representations `H ∈ ℝᴺˣᵈ`.
2. **Task-Conditioned Cross-Attention** — A learnable task identity embedding `pₜ` interacts with a near-zero noise input matrix `X_noise ~ N(0, σ²I)` (with `σ = 0.01`) through multi-head cross-attention, producing a task-conditioned prompt vector `zₜ`.
3. **Residual Prompt Injection** — The prompt `zₜ` is projected and added to the molecule-level embedding as a residual shift before the prediction head: `H' = H + α · W_proj(zₜ)`, where `α` is a small mixing coefficient.

---

## Environment Setup

**Option 1: Conda (recommended)**

```bash
conda env create -f environment.yaml
conda activate tapt
```

**Option 2: pip**

```bash
pip install -r requirements.txt
bash install.sh
```

> **Note:** Place the pre-trained graph encoder weights at:
> `./dumped/pretrained_graph_encoder/original_CMPN_0623_1350_14000th_epoch.pkl`

---

## Quick Start

The main entry point is `train.py`, which supports two modes:

- **TAPT (proposed)** — enabled with `--use_tapt`
- **KG-Augmented Baseline (KANO)** — default behavior (without `--use_tapt`)

Both modes share the same CMPNN backbone and the same pre-trained checkpoint.

### Pre-training (optional)

```bash
python pretrain.py
```

---

## Benchmarks

The commands below reproduce the main results reported in the paper. All commands assume the pre-trained CMPNN checkpoint is at `./dumped/pretrained_graph_encoder/original_CMPN_0623_1350_14000th_epoch.pkl`.

### Classification Tasks

#### BBBP

```bash
python train.py \
    --use_tapt \
    --data_path ./data/bbbp.csv \
    --metric auc \
    --dataset_type classification \
    --epochs 100 \
    --num_runs 3 \
    --ensemble_size 3 \
    --gpu 0 \
    --batch_size 50 \
    --seed 4 \
    --init_lr 1e-4 --max_lr 1e-3 --final_lr 1e-4 \
    --warmup_epochs 2.0 \
    --split_type scaffold_balanced \
    --exp_name bbbp_tapt \
    --exp_id bbbp_tapt \
    --checkpoint_path ./dumped/pretrained_graph_encoder/original_CMPN_0623_1350_14000th_epoch.pkl \
    --prompt_dim 256 \
    --prompt_lr 1e-4 \
    --tapt_alpha 0.1 \
    --structure_noise_scale 0.01
```

#### Tox21

```bash
python train.py \
    --use_tapt \
    --data_path ./data/tox21.csv \
    --metric auc \
    --dataset_type classification \
    --epochs 50 \
    --num_runs 3 \
    --ensemble_size 3 \
    --gpu 0 \
    --batch_size 50 \
    --seed 43 \
    --init_lr 1e-4 --max_lr 1e-3 --final_lr 1e-4 \
    --warmup_epochs 2.0 \
    --split_type scaffold_balanced \
    --exp_name tox21_tapt \
    --exp_id tox21_tapt \
    --checkpoint_path ./dumped/pretrained_graph_encoder/original_CMPN_0623_1350_14000th_epoch.pkl \
    --prompt_dim 256 \
    --prompt_lr 1e-4 \
    --tapt_alpha 0.1 \
    --structure_noise_scale 0.01
```

#### ToxCast

```bash
python train.py \
    --use_tapt \
    --data_path ./data/toxcast.csv \
    --dataset_type classification \
    --metric auc \
    --epochs 100 \
    --num_runs 3 \
    --ensemble_size 3 \
    --gpu 0 \
    --batch_size 50 \
    --seed 43 \
    --init_lr 1e-4 --max_lr 1e-3 --final_lr 1e-4 \
    --warmup_epochs 2.0 \
    --split_type scaffold_balanced \
    --exp_name toxcast_tapt \
    --exp_id toxcast_tapt \
    --checkpoint_path ./dumped/pretrained_graph_encoder/original_CMPN_0623_1350_14000th_epoch.pkl \
    --prompt_dim 256 \
    --prompt_lr 1e-4 \
    --tapt_alpha 0.1 \
    --structure_noise_scale 0.01
```

#### SIDER

```bash
python train.py \
    --use_tapt \
    --data_path ./data/sider.csv \
    --metric auc \
    --dataset_type classification \
    --epochs 100 \
    --num_runs 3 \
    --ensemble_size 3 \
    --gpu 0 \
    --batch_size 50 \
    --seed 43 \
    --init_lr 1e-4 --max_lr 1e-3 --final_lr 1e-4 \
    --warmup_epochs 2.0 \
    --split_type scaffold_balanced \
    --exp_name sider_tapt \
    --exp_id sider_tapt \
    --checkpoint_path ./dumped/pretrained_graph_encoder/original_CMPN_0623_1350_14000th_epoch.pkl \
    --prompt_dim 256 \
    --prompt_lr 1e-4 \
    --tapt_alpha 0.1 \
    --structure_noise_scale 0.01
```

#### ClinTox

```bash
python train.py \
    --use_tapt \
    --data_path ./data/clintox.csv \
    --dataset_type classification \
    --metric auc \
    --epochs 100 \
    --num_runs 3 \
    --ensemble_size 3 \
    --gpu 0 \
    --batch_size 50 \
    --seed 43 \
    --init_lr 1e-4 --max_lr 1e-3 --final_lr 1e-4 \
    --warmup_epochs 2.0 \
    --split_type scaffold_balanced \
    --exp_name clintox_tapt \
    --exp_id clintox_tapt \
    --checkpoint_path ./dumped/pretrained_graph_encoder/original_CMPN_0623_1350_14000th_epoch.pkl \
    --prompt_dim 256 \
    --prompt_lr 1e-4 \
    --tapt_alpha 0.1 \
    --structure_noise_scale 0.01
```

### Regression Tasks

#### ESOL

```bash
python train.py \
    --use_tapt \
    --data_path ./data/esol.csv \
    --dataset_type regression \
    --metric rmse \
    --epochs 100 \
    --num_runs 3 \
    --ensemble_size 3 \
    --gpu 0 \
    --batch_size 32 \
    --seed 4 \
    --init_lr 1e-4 --max_lr 1e-3 --final_lr 1e-4 \
    --warmup_epochs 2.0 \
    --split_type scaffold_balanced \
    --exp_name esol_tapt \
    --exp_id esol_tapt \
    --checkpoint_path ./dumped/pretrained_graph_encoder/original_CMPN_0623_1350_14000th_epoch.pkl \
    --prompt_dim 128 \
    --prompt_lr 5e-5 \
    --tapt_alpha 0.1 \
    --structure_noise_scale 0.01
```

#### FreeSolv

```bash
python train.py \
    --use_tapt \
    --data_path ./data/freesolv.csv \
    --dataset_type regression \
    --metric rmse \
    --epochs 100 \
    --num_runs 3 \
    --ensemble_size 3 \
    --gpu 0 \
    --batch_size 10 \
    --seed 10 \
    --init_lr 5e-5 --max_lr 5e-4 --final_lr 5e-5 \
    --warmup_epochs 2.0 \
    --split_type scaffold_balanced \
    --exp_name freesolv_tapt \
    --exp_id freesolv_tapt \
    --checkpoint_path ./dumped/pretrained_graph_encoder/original_CMPN_0623_1350_14000th_epoch.pkl \
    --prompt_dim 128 \
    --prompt_lr 5e-5 \
    --tapt_alpha 0.1 \
    --structure_noise_scale 0.01
```

#### Lipo

```bash
python train.py \
    --use_tapt \
    --data_path ./data/lipo.csv \
    --metric rmse \
    --dataset_type regression \
    --epochs 100 \
    --num_runs 3 \
    --ensemble_size 3 \
    --gpu 0 \
    --batch_size 50 \
    --seed 43 \
    --init_lr 1e-4 --max_lr 1e-3 --final_lr 1e-4 \
    --warmup_epochs 2.0 \
    --split_type scaffold_balanced \
    --exp_name lipo_tapt \
    --exp_id lipo_tapt \
    --checkpoint_path ./dumped/pretrained_graph_encoder/original_CMPN_0623_1350_14000th_epoch.pkl \
    --prompt_dim 256 \
    --prompt_lr 1e-4 \
    --tapt_alpha 0.1 \
    --structure_noise_scale 0.01
```

#### QM7

```bash
python train.py \
    --use_tapt \
    --data_path ./data/qm7.csv \
    --dataset_type regression \
    --metric mae \
    --epochs 100 \
    --num_runs 3 \
    --ensemble_size 3 \
    --gpu 0 \
    --batch_size 32 \
    --seed 4 \
    --init_lr 1e-4 --max_lr 1e-3 --final_lr 1e-4 \
    --warmup_epochs 2.0 \
    --split_type scaffold_balanced \
    --exp_name qm7_tapt \
    --exp_id qm7_tapt \
    --checkpoint_path ./dumped/pretrained_graph_encoder/original_CMPN_0623_1350_14000th_epoch.pkl \
    --prompt_dim 128 \
    --prompt_lr 5e-5 \
    --tapt_alpha 0.1 \
    --structure_noise_scale 0.01
```

#### QM8

```bash
python train.py \
    --use_tapt \
    --data_path ./data/qm8.csv \
    --dataset_type regression \
    --metric mae \
    --epochs 100 \
    --num_runs 3 \
    --ensemble_size 3 \
    --gpu 0 \
    --batch_size 32 \
    --seed 4 \
    --init_lr 1e-4 --max_lr 1e-3 --final_lr 1e-4 \
    --warmup_epochs 2.0 \
    --split_type scaffold_balanced \
    --exp_name qm8_tapt \
    --exp_id qm8_tapt \
    --checkpoint_path ./dumped/pretrained_graph_encoder/original_CMPN_0623_1350_14000th_epoch.pkl \
    --prompt_dim 128 \
    --prompt_lr 5e-5 \
    --tapt_alpha 0.1 \
    --structure_noise_scale 0.01
```

---

## Reproducing the KG-Augmented Baseline (KANO)

To reproduce the KANO baseline reported in the paper, run the same commands as above **without** the `--use_tapt`, `--tapt_alpha`, and `--structure_noise_scale` flags.

#### ESOL — KANO baseline

```bash
python train.py \
    --data_path ./data/esol.csv \
    --dataset_type regression \
    --metric rmse \
    --epochs 100 \
    --num_runs 3 \
    --ensemble_size 3 \
    --gpu 0 \
    --batch_size 32 \
    --seed 4 \
    --init_lr 1e-4 --max_lr 1e-3 --final_lr 1e-4 \
    --warmup_epochs 2.0 \
    --split_type scaffold_balanced \
    --exp_name esol_kano \
    --exp_id esol_kano \
    --checkpoint_path ./dumped/pretrained_graph_encoder/original_CMPN_0623_1350_14000th_epoch.pkl \
    --prompt_dim 128 \
    --prompt_lr 5e-5
```

#### Tox21 — KANO baseline

```bash
python train.py \
    --data_path ./data/tox21.csv \
    --metric auc \
    --dataset_type classification \
    --epochs 50 \
    --num_runs 3 \
    --ensemble_size 3 \
    --gpu 0 \
    --batch_size 50 \
    --seed 43 \
    --init_lr 1e-4 --max_lr 1e-3 --final_lr 1e-4 \
    --warmup_epochs 2.0 \
    --split_type scaffold_balanced \
    --exp_name tox21_kano \
    --exp_id tox21_kano \
    --checkpoint_path ./dumped/pretrained_graph_encoder/original_CMPN_0623_1350_14000th_epoch.pkl \
    --prompt_dim 256 \
    --prompt_lr 1e-4
```

The remaining KANO commands follow the same pattern: drop `--use_tapt`, `--tapt_alpha`, and `--structure_noise_scale` from the corresponding TAPT command.

---

## Key Arguments

| Argument | Description | Default |
|---|---|---|
| `--use_tapt` | Enable TAPT prompt tuning. Without this flag, the KG-augmented (KANO) baseline is used. | `False` |
| `--dataset_type` | `classification`, `regression`, or `multiclass` | — |
| `--metric` | `auc`, `prc-auc`, `rmse`, `mae`, `mse`, `r2`, `accuracy`, `cross_entropy` | — |
| `--seed` | Random seed; use `-1` for a random seed each run | `0` |
| `--num_runs` | Number of independent training runs | `1` |
| `--ensemble_size` | Number of models in the ensemble | `1` |
| `--prompt_dim` | Prompt embedding dimension (TAPT-only) | `128` |
| `--prompt_lr` | Learning rate for prompt parameters | `1e-3` |
| `--backbone_lr` | Learning rate for backbone encoder parameters | `1e-5` |
| `--tapt_alpha` | Mixing coefficient for the TAPT prompt residual | `0.001` |
| `--structure_noise_scale` | Standard deviation σ of the near-zero noise K/V input | `0.01` |
| `--num_prompt_tokens` | Number of learnable prompt tokens per task | `5` |
| `--tapt_dropout` | Dropout rate inside the TAPT module | `0.1` |
| `--split_type` | Data split strategy (e.g., `scaffold_balanced`) | `random` |
| `--init_lr` / `--max_lr` / `--final_lr` | Learning rate schedule (NoamLR) | `1e-4` / `1e-3` / `1e-4` |
| `--warmup_epochs` | Warmup epochs for NoamLR | `2.0` |
| `--checkpoint_path` | Path to the pre-trained CMPNN checkpoint | — |

---

## Project Structure

```
TAPT-main/
├── chemprop/                       # Core model and training library
│   ├── models/
│   │   ├── model.py                # MoleculeModel and TAPT injection logic
│   │   ├── tapt_modules.py         # TAPTPromptModule (cross-attention)
│   │   ├── cmpn.py                 # CMPNN encoder
│   │   └── mpn.py                  # MPN encoder (alternative)
│   ├── train/
│   │   ├── run_training.py         # Main training loop
│   │   ├── train.py                # Inner per-epoch training step
│   │   ├── evaluate.py             # Validation/test evaluation
│   │   └── predict.py              # Inference
│   ├── data/                       # Data loading and splitting
│   ├── features/                   # Molecular featurization
│   └── utils.py                    # Optimizer, scheduler, checkpoint utilities
├── data/                           # Benchmark datasets (CSV)
├── dumped/                         # Checkpoints and pre-trained weights
├── logs/                           # Training logs
├── train.py                        # Main training entry point
├── pretrain.py                     # Pre-training script
├── finetune.sh                     # Example fine-tuning command
├── install.sh                      # Dependency installation script
├── environment.yaml                # Conda environment specification
└── requirements.txt                # pip requirements
```

## Acknowledgements

This repository builds upon [Chemprop](https://github.com/chemprop/chemprop), [CMPNN](https://github.com/SY575/CMPNN), and [KANO](https://github.com/HICAI-ZJU/KANO). We thank the authors for open-sourcing their implementations.