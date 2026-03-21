#!/bin/bash
# TAPT 框架运行脚本（严格适配指定命令参数）
# 用途：一键执行11个分子属性预测任务，参数完全对齐指定命令
# 使用前激活环境：conda activate tapt
# 注释掉不需要运行的任务块即可

# ======================== 全局核心配置（完全对齐指定命令）=======================
BASE_DATA_PATH="./data"
CHECKPOINT_PATH="./dumped/pretrained_graph_encoder/original_CMPN_0623_1350_14000th_epoch.pkl"
GPU_ID=0
SEED=4
EPOCHS=100
NUM_RUNS=3
ENSEMBLE_SIZE=1  # 严格按指定命令设为1
BATCH_SIZE=50    # 严格按指定命令设为50
WARMUP_EPOCHS=2.0
INIT_LR=1e-4
MAX_LR=1e-3
FINAL_LR=1e-4
PROMPT_DIM=256   # 严格按指定命令设为256
PROMPT_LR=1e-4   # 严格按指定命令设为1e-4
USE_TAPT="--use_tapt"
MODE="mode1"     # 新增指定命令中的mode1参数

# ======================== 分类任务 (Classification Tasks) ========================



# ======================== 回归任务 (Regression Tasks) ========================

## 7. ESOL 数据集
echo -e "\n===== 开始运行 ESOL 回归任务 ====="
python train.py \
    ${USE_TAPT} \
    --mode ${MODE} \
    --data_path ${BASE_DATA_PATH}/esol.csv \
    --metric r2 \
    --dataset_type regression \
    --epochs ${EPOCHS} \
    --num_runs ${NUM_RUNS} \
    --ensemble_size ${ENSEMBLE_SIZE} \
    --gpu ${GPU_ID} \
    --batch_size ${BATCH_SIZE} \
    --seed ${SEED} \
    --init_lr ${INIT_LR} --max_lr ${MAX_LR} --final_lr ${FINAL_LR} \
    --warmup_epochs ${WARMUP_EPOCHS} \
    --split_type scaffold_balanced \
    --exp_name esol_kano_fixed \
    --exp_id esol_kano_fixed \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --prompt_dim ${PROMPT_DIM} \
    --prompt_lr ${PROMPT_LR}

## 8. FreeSolv 数据集
echo -e "\n===== 开始运行 FreeSolv 回归任务 ====="
python train.py \
    ${USE_TAPT} \
    --mode ${MODE} \
    --data_path ${BASE_DATA_PATH}/freesolv.csv \
    --metric r2 \
    --dataset_type regression \
    --epochs ${EPOCHS} \
    --num_runs ${NUM_RUNS} \
    --ensemble_size ${ENSEMBLE_SIZE} \
    --gpu ${GPU_ID} \
    --batch_size ${BATCH_SIZE} \
    --seed ${SEED} \
    --init_lr ${INIT_LR} --max_lr ${MAX_LR} --final_lr ${FINAL_LR} \
    --warmup_epochs ${WARMUP_EPOCHS} \
    --split_type scaffold_balanced \
    --exp_name freesolv_kano_fixed \
    --exp_id freesolv_kano_fixed \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --prompt_dim ${PROMPT_DIM} \
    --prompt_lr ${PROMPT_LR}

## 9. Lipo 数据集
echo -e "\n===== 开始运行 Lipo 回归任务 ====="
python train.py \
    ${USE_TAPT} \
    --mode ${MODE} \
    --data_path ${BASE_DATA_PATH}/lipo.csv \
    --metric r2 \
    --dataset_type regression \
    --epochs ${EPOCHS} \
    --num_runs ${NUM_RUNS} \
    --ensemble_size ${ENSEMBLE_SIZE} \
    --gpu ${GPU_ID} \
    --batch_size ${BATCH_SIZE} \
    --seed ${SEED} \
    --init_lr ${INIT_LR} --max_lr ${MAX_LR} --final_lr ${FINAL_LR} \
    --warmup_epochs ${WARMUP_EPOCHS} \
    --split_type scaffold_balanced \
    --exp_name lipo_kano_fixed \
    --exp_id lipo_kano_fixed \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --prompt_dim ${PROMPT_DIM} \
    --prompt_lr ${PROMPT_LR}

## 10. QM7 数据集
echo -e "\n===== 开始运行 QM7 回归任务 ====="
python train.py \
    ${USE_TAPT} \
    --mode ${MODE} \
    --data_path ${BASE_DATA_PATH}/qm7.csv \
    --metric mae \
    --dataset_type regression \
    --epochs ${EPOCHS} \
    --num_runs ${NUM_RUNS} \
    --ensemble_size ${ENSEMBLE_SIZE} \
    --gpu ${GPU_ID} \
    --batch_size ${BATCH_SIZE} \
    --seed ${SEED} \
    --init_lr ${INIT_LR} --max_lr ${MAX_LR} --final_lr ${FINAL_LR} \
    --warmup_epochs ${WARMUP_EPOCHS} \
    --split_type random \  # QM7无scaffold，适配random划分
    --exp_name qm7_kano_fixed \
    --exp_id qm7_kano_fixed \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --prompt_dim ${PROMPT_DIM} \
    --prompt_lr ${PROMPT_LR}

## 11. QM8 数据集
echo -e "\n===== 开始运行 QM8 回归任务 ====="
python train.py \
    ${USE_TAPT} \
    --mode ${MODE} \
    --data_path ${BASE_DATA_PATH}/qm8.csv \
    --metric mae \
    --dataset_type regression \
    --epochs ${EPOCHS} \
    --num_runs ${NUM_RUNS} \
    --ensemble_size ${ENSEMBLE_SIZE} \
    --gpu ${GPU_ID} \
    --batch_size ${BATCH_SIZE} \
    --seed ${SEED} \
    --init_lr ${INIT_LR} --max_lr ${MAX_LR} --final_lr ${FINAL_LR} \
    --warmup_epochs ${WARMUP_EPOCHS} \
    --split_type random \  # QM8无scaffold，适配random划分
    --exp_name qm8_kano_fixed \
    --exp_id qm8_kano_fixed \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --prompt_dim ${PROMPT_DIM} \
    --prompt_lr ${PROMPT_LR}

echo -e "\n===== 所有指定任务运行完成 ====="