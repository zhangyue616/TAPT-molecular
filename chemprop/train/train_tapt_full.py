# ===== train_tapt_full.py (完整 4 模块 TAPT 训练脚本) =====
"""
TAPT Full (4-Module) 训练脚本
支持完整 4 模块 TAPT 和 Ablation 实验

修改历史:
- 2026-01-23: 创建完整 4 模块 TAPT 训练脚本
- 2026-01-23: 添加 Gate 监控和 Ablation 支持
"""

from argparse import Namespace
import csv
from logging import Logger
import os
from typing import List, Dict
import json

import numpy as np
from tensorboardX import SummaryWriter
import torch
import pickle
from torch.optim.lr_scheduler import ExponentialLR

from chemprop.train.evaluate import evaluate, evaluate_predictions
from chemprop.train.predict import predict
from chemprop.train.train import train
from chemprop.data import StandardScaler
from chemprop.data.utils import get_class_sizes, get_data, get_task_names, split_data
from chemprop.models import build_model
from chemprop.nn_utils import param_count
from chemprop.utils import build_optimizer, build_lr_scheduler, get_loss_func, get_metric_func, \
    load_checkpoint, makedirs, save_checkpoint


def run_training_tapt_full(args: Namespace, logger: Logger = None) -> List[float]:
    """
    训练完整 4 模块 TAPT 模型

    :param args: 训练参数
    :param logger: 日志记录器
    :return: 每个 run 的 ensemble 分数列表
    """
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Set GPU
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    # ============================================================
    # 数据加载与划分（只做一次）
    # ============================================================
    info('Loading data')
    args.task_names = get_task_names(args.data_path)
    data = get_data(path=args.data_path, args=args, logger=logger)
    args.num_tasks = data.num_tasks()
    args.features_size = data.features_size()
    info(f'Number of tasks = {args.num_tasks}')

    # Split data
    debug(f'Splitting data with seed {args.seed}')

    if args.separate_test_path:
        test_data = get_data(path=args.separate_test_path, args=args,
                             features_path=args.separate_test_features_path,
                             logger=logger)
    if args.separate_val_path:
        val_data = get_data(path=args.separate_val_path, args=args,
                            features_path=args.separate_val_features_path,
                            logger=logger)

    if args.separate_val_path and args.separate_test_path:
        train_data = data
    elif args.separate_val_path:
        train_data, _, test_data = split_data(data=data, split_type=args.split_type,
                                              sizes=(0.8, 0.2, 0.0),
                                              seed=args.seed, args=args, logger=logger)
    elif args.separate_test_path:
        train_data, val_data, _ = split_data(data=data, split_type=args.split_type,
                                             sizes=(0.8, 0.2, 0.0),
                                             seed=args.seed, args=args, logger=logger)
    else:
        train_data, val_data, test_data = split_data(data=data, split_type=args.split_type,
                                                     sizes=args.split_sizes,
                                                     seed=args.seed, args=args, logger=logger)

    if args.dataset_type == 'classification':
        class_sizes = get_class_sizes(data)
        debug('Class sizes')
        for i, task_class_sizes in enumerate(class_sizes):
            debug(f'{args.task_names[i]} '
                  f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')

    # Features scaling
    if args.features_scaling:
        features_scaler = train_data.normalize_features(replace_nan_token=0)
        val_data.normalize_features(features_scaler)
        test_data.normalize_features(features_scaler)
    else:
        features_scaler = None

    args.train_data_size = len(train_data)

    debug(f'Total size = {len(data):,} | '
          f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')

    # Initialize scaler
    if args.dataset_type == 'regression':
        debug('Fitting scaler')
        train_smiles, train_targets = train_data.smiles(), train_data.targets()
        scaler = StandardScaler().fit(train_targets)
        scaled_targets = scaler.transform(train_targets).tolist()
        train_data.set_targets(scaled_targets)
    else:
        scaler = None

    # Get loss and metric functions
    loss_func = get_loss_func(args)
    metric_func = get_metric_func(metric=args.metric)

    # Set up test set evaluation
    test_smiles, test_targets = test_data.smiles(), test_data.targets()

    # ============================================================
    # 🚀 外层循环 - num_runs
    # ============================================================
    all_run_scores = []
    all_gate_histories = []  # 记录所有 runs 的 gate 变化

    for run_idx in range(args.num_runs):
        info('\n' + '=' * 80)
        info(f'🚀 Starting Run {run_idx + 1}/{args.num_runs} (seed={args.seed})')
        info('=' * 80 + '\n')

        # 初始化预测累加器
        if args.dataset_type == 'multiclass':
            sum_test_preds = np.zeros((len(test_smiles), args.num_tasks, args.multiclass_num_classes))
        else:
            sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))

        run_gate_history = []  # 当前 run 的 gate 历史

        # ============================================================
        # 内层循环 - Ensemble
        # ============================================================
        for model_idx in range(args.ensemble_size):
            info(f'\n📦 Training Model {model_idx + 1}/{args.ensemble_size} (Run {run_idx + 1}/{args.num_runs})\n')

            # 设置保存目录
            save_dir = os.path.join(args.save_dir, f'run_{run_idx}', f'model_{model_idx}')
            makedirs(save_dir)

            try:
                writer = SummaryWriter(log_dir=save_dir)
            except:
                writer = SummaryWriter(logdir=save_dir)

            # ============================================================
            # 🚀 构建 TAPT Full 模型
            # ============================================================
            info('🚀 Building TAPT Full (4-module) model...')

            # 导入 TAPT Full 模块
            from chemprop.models.model_tapt_full import (
                build_tapt_full_model,
                freeze_kano_only,
                freeze_specific_modules,
                get_tapt_parameter_groups
            )

            # 1. 先构建基础 KANO 模型
            base_model = build_model(args, encoder_name=getattr(args, 'encoder_name', 'CMPNN'))

            # 2. 加载 KANO 预训练权重
            if args.checkpoint_path is not None:
                info(f'Loading KANO checkpoint: {args.checkpoint_path}')
                state_dict = torch.load(args.checkpoint_path, map_location='cpu')

                # 加载 encoder
                encoder_state = state_dict.get('encoder', state_dict)
                missing_keys, unexpected_keys = base_model.encoder.load_state_dict(encoder_state, strict=False)

                loaded_params = sum(p.numel() for p in base_model.encoder.parameters())
                info(f'✅ KANO encoder weights loaded: {loaded_params:,} parameters')

                if missing_keys:
                    debug(f'⚠️  Missing keys: {len(missing_keys)}')
                if unexpected_keys:
                    debug(f'⚠️  Unexpected keys: {len(unexpected_keys)}')

                # 加载 FFN
                if 'ffn' in state_dict and hasattr(base_model, 'ffn'):
                    try:
                        base_model.ffn.load_state_dict(state_dict['ffn'], strict=False)
                        ffn_params = sum(p.numel() for p in base_model.ffn.parameters())
                        info(f'✅ KANO FFN weights loaded: {ffn_params:,} parameters')
                    except Exception as e:
                        debug(f'⚠️  Warning: Could not load FFN weights: {e}')

            # 3. 构建 TAPT Full 模型（包装 base_model）
            model = build_tapt_full_model(args, base_model)

            # 4. 冻结 KANO（如果需要）
            if getattr(args, 'freeze_kano', False):
                info('🔒 Freezing KANO parameters...')
                kano_frozen, tapt_trainable = freeze_kano_only(model)
                info(f'❄️  KANO frozen: {kano_frozen:,} parameters')
                info(f'✅ TAPT trainable: {tapt_trainable:,} parameters')

            # 5. 冻结特定模块（如果需要）
            if getattr(args, 'freeze_task_prompt', False) or \
                    getattr(args, 'freeze_struct_prompt', False) or \
                    getattr(args, 'freeze_pyramid_agg', False) or \
                    getattr(args, 'freeze_node_injection', False):
                info('🔒 Freezing specific TAPT modules...')
                freeze_specific_modules(
                    model,
                    freeze_task=getattr(args, 'freeze_task_prompt', False),
                    freeze_struct=getattr(args, 'freeze_struct_prompt', False),
                    freeze_pyramid=getattr(args, 'freeze_pyramid_agg', False),
                    freeze_node=getattr(args, 'freeze_node_injection', False)
                )

            # 打印模型配置
            info(f'✅ TAPT Full model built:')
            info(f'  - Task Prompt: {getattr(args, "enable_task_prompt", True)}')
            info(f'  - Struct Prompt: {getattr(args, "enable_struct_prompt", True)}')
            info(f'  - Pyramid Agg: {getattr(args, "enable_pyramid_agg", True)}')
            info(f'  - Node Injection: {getattr(args, "enable_node_injection", True)}')

            debug(model)
            debug(f'Number of parameters = {param_count(model):,}')

            if args.cuda:
                debug('Moving model to cuda')
                model = model.cuda()

            # Save initial model
            save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)

            # ============================================================
            # Optimizer
            # ============================================================
            if getattr(args, 'freeze_kano', False):
                info('🎯 Optimizer mode: TAPT parameters only')

                tapt_params = list(model.tapt_module.parameters())

                if not tapt_params:
                    raise ValueError("No TAPT parameters found!")

                trainable_tapt_params = sum(p.numel() for p in tapt_params if p.requires_grad)
                info(f'📦 TAPT trainable parameters: {trainable_tapt_params:,}')

                if trainable_tapt_params == 0:
                    raise RuntimeError("❌ No trainable TAPT parameters!")

                optimizer = torch.optim.Adam(
                    tapt_params,
                    lr=getattr(args, 'tapt_lr', 1e-5),
                    weight_decay=getattr(args, 'weight_decay', 1e-4)
                )
                info(f'✅ Optimizer: Adam (TAPT only, lr={getattr(args, "tapt_lr", 1e-5)})')

            else:
                info('🎯 Optimizer mode: Differential learning rates')

                kano_lr = getattr(args, 'kano_lr', 5e-5)
                tapt_lr = getattr(args, 'tapt_lr', 1e-4)

                param_groups = get_tapt_parameter_groups(model, kano_lr, tapt_lr)

                optimizer = torch.optim.Adam(
                    param_groups,
                    weight_decay=getattr(args, 'weight_decay', 1e-4)
                )
                info(f'✅ Optimizer: Adam (KANO lr={kano_lr}, TAPT lr={tapt_lr})')

            # ============================================================
            # Scheduler
            # ============================================================
            if getattr(args, 'freeze_kano', False):
                info('📊 Scheduler: ExponentialLR (gamma=0.99)')
                scheduler = ExponentialLR(optimizer, gamma=0.99)
            else:
                info('📊 Scheduler: NoamLR (differential LR)')
                scheduler = build_lr_scheduler(optimizer, args)

            # ============================================================
            # 打印模型信息（第一个 model）
            # ============================================================
            if model_idx == 0 and run_idx == 0:
                info('\n' + '=' * 80)
                info('📊 MODEL ARCHITECTURE SUMMARY')
                info('=' * 80)

                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                frozen_params = total_params - trainable_params

                info(f'Total parameters: {total_params:,}')
                info(f'Trainable parameters: {trainable_params:,} ({trainable_params / total_params * 100:.2f}%)')
                info(f'Frozen parameters: {frozen_params:,} ({frozen_params / total_params * 100:.2f}%)')

                tapt_params = sum(p.numel() for p in model.tapt_module.parameters() if p.requires_grad)
                info(f'TAPT parameters: {tapt_params:,} ({tapt_params / trainable_params * 100:.2f}% of trainable)')

                info('=' * 80 + '\n')

            # ============================================================
            # Training Loop
            # ============================================================
            best_score = float('inf') if args.minimize_score else -float('inf')
            best_epoch, n_iter = 0, 0

            model_gate_history = {
                'task': [],
                'struct': [],
                'pyramid': [],
                'node': []
            }

            for epoch in range(args.epochs):
                info(f'Epoch {epoch}')

                # Train
                n_iter = train(
                    model=model,
                    prompt=True,  # TAPT 使用 prompt
                    data=train_data,
                    loss_func=loss_func,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    args=args,
                    n_iter=n_iter,
                    logger=logger,
                    writer=writer
                )

                if isinstance(scheduler, ExponentialLR):
                    scheduler.step()

                # Validation
                val_scores = evaluate(
                    model=model,
                    prompt=True,
                    data=val_data,
                    num_tasks=args.num_tasks,
                    metric_func=metric_func,
                    batch_size=args.batch_size,
                    dataset_type=args.dataset_type,
                    scaler=scaler,
                    logger=logger
                )

                avg_val_score = np.nanmean(val_scores)
                info(f'Validation {args.metric} = {avg_val_score:.6f}')
                writer.add_scalar(f'validation_{args.metric}', avg_val_score, n_iter)

                # Test (for monitoring)
                test_preds = predict(
                    model=model,
                    prompt=True,
                    data=test_data,
                    batch_size=args.batch_size,
                    scaler=scaler
                )
                test_scores = evaluate_predictions(
                    preds=test_preds,
                    targets=test_targets,
                    num_tasks=args.num_tasks,
                    metric_func=metric_func,
                    dataset_type=args.dataset_type,
                    logger=logger
                )

                avg_test_score = np.nanmean(test_scores)
                info(f'Test {args.metric} = {avg_test_score:.6f}')

                # ============================================================
                # 🔍 记录 Gate 值（每 10 epochs）
                # ============================================================
                if epoch % 10 == 0:
                    model.eval()
                    with torch.no_grad():
                        # 取一个 batch 获取 gates
                        sample_batch = val_data[:min(10, len(val_data))]
                        sample_mol_data = sample_batch.batch_graph()

                        if args.cuda:
                            sample_mol_data = sample_mol_data.cuda()

                        _, gates = model(sample_mol_data, None)

                        gate_task = gates['task'].item() if torch.is_tensor(gates['task']) else gates['task']
                        gate_struct = gates['struct'].item() if torch.is_tensor(gates['struct']) else gates['struct']
                        gate_pyramid = gates['pyramid'].item() if torch.is_tensor(gates['pyramid']) else gates[
                            'pyramid']
                        gate_node = gates['node'].item() if torch.is_tensor(gates['node']) else gates['node']

                        model_gate_history['task'].append(gate_task)
                        model_gate_history['struct'].append(gate_struct)
                        model_gate_history['pyramid'].append(gate_pyramid)
                        model_gate_history['node'].append(gate_node)

                        info(f'📊 Gate values at epoch {epoch}:')
                        info(f'  - Task: {gate_task:.6f}')
                        info(f'  - Struct: {gate_struct:.6f}')
                        info(f'  - Pyramid: {gate_pyramid:.6f}')
                        info(f'  - Node: {gate_node:.6f}')

                        # 写入 TensorBoard
                        writer.add_scalar('gate/task', gate_task, epoch)
                        writer.add_scalar('gate/struct', gate_struct, epoch)
                        writer.add_scalar('gate/pyramid', gate_pyramid, epoch)
                        writer.add_scalar('gate/node', gate_node, epoch)

                    model.train()

                # Save best model
                if args.minimize_score and avg_val_score < best_score or \
                        not args.minimize_score and avg_val_score > best_score:
                    best_score, best_epoch = avg_val_score, epoch
                    save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)

            # ============================================================
            # 评估最佳模型
            # ============================================================
            info(f'Model {model_idx} best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')
            model = load_checkpoint(os.path.join(save_dir, 'model.pt'), cuda=args.cuda, logger=logger)

            test_preds = predict(
                model=model,
                prompt=True,
                data=test_data,
                batch_size=args.batch_size,
                scaler=scaler
            )
            test_scores = evaluate_predictions(
                preds=test_preds,
                targets=test_targets,
                num_tasks=args.num_tasks,
                metric_func=metric_func,
                dataset_type=args.dataset_type,
                logger=logger
            )

            if len(test_preds) != 0:
                sum_test_preds += np.array(test_preds)

            avg_test_score = np.nanmean(test_scores)
            info(f'Model {model_idx} test {args.metric} = {avg_test_score:.6f}')
            writer.add_scalar(f'test_{args.metric}', avg_test_score, 0)

            # 保存 gate history
            gate_history_path = os.path.join(save_dir, 'gate_history.json')
            with open(gate_history_path, 'w') as f:
                json.dump(model_gate_history, f, indent=2)
            info(f'✅ Gate history saved to {gate_history_path}')

            run_gate_history.append(model_gate_history)

        # ============================================================
        # 评估 Ensemble
        # ============================================================
        avg_test_preds = (sum_test_preds / args.ensemble_size).tolist()

        ensemble_scores = evaluate_predictions(
            preds=avg_test_preds,
            targets=test_targets,
            num_tasks=args.num_tasks,
            metric_func=metric_func,
            dataset_type=args.dataset_type,
            logger=logger
        )

        avg_ensemble_test_score = np.nanmean(ensemble_scores)
        info(f'\n🎯 Run {run_idx + 1}/{args.num_runs} Ensemble test {args.metric} = {avg_ensemble_test_score:.6f}')

        all_run_scores.append(avg_ensemble_test_score)
        all_gate_histories.append(run_gate_history)

    # ============================================================
    # 最终统计
    # ============================================================
    if args.num_runs > 1:
        mean_score = np.mean(all_run_scores)
        std_score = np.std(all_run_scores, ddof=1)

        info('\n' + '=' * 80)
        info('📊 FINAL RESULTS ACROSS ALL RUNS')
        info('=' * 80)
        info(f'Individual run scores: {[f"{s:.6f}" for s in all_run_scores]}')
        info(f'Mean ± Std: {mean_score:.6f} ± {std_score:.6f}')
        info(f'\n✅ Final Test {args.metric} = {mean_score:.3f} ± {std_score:.3f}')
        info('=' * 80 + '\n')

        # 保存所有 gate histories
        gate_summary_path = os.path.join(args.save_dir, 'all_gate_histories.json')
        with open(gate_summary_path, 'w') as f:
            json.dump(all_gate_histories, f, indent=2)
        info(f'✅ All gate histories saved to {gate_summary_path}')

        return all_run_scores
    else:
        info(f'\n✅ Final Test {args.metric} = {all_run_scores[0]:.6f}')
        return [all_run_scores[0]]


# ============================================================
# Main Entry Point
# ============================================================
if __name__ == '__main__':
    from chemprop.args import TrainArgs
    from chemprop.train import cross_validate
    from chemprop.utils import create_logger

    args = TrainArgs().parse_args()
    logger = create_logger(name='train_tapt_full', save_dir=args.save_dir, quiet=args.quiet)

    run_training_tapt_full(args, logger)
