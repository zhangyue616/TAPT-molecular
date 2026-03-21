# ===== chemprop/train/run_training.py (TAPT 完全集成版) =====
"""
KANO/TAPT 训练主函数
支持原版 KANO 和新增 TAPT 模式

修改历史:
- 2026-01-20: 集成 TAPT 支持
- 2026-01-20: 添加 num_runs 外层循环支持
- 2026-01-22: 修复数据划分问题 - 所有 runs 使用相同数据
- 2026-01-23: 完全集成 model.py 中的 TAPT 组件
"""
import logging
from argparse import Namespace
import csv
from logging import Logger
import os
from typing import List

import numpy as np
from tensorboardX import SummaryWriter
import torch
import pickle
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from .evaluate import evaluate, evaluate_predictions
from .predict import predict
from .train import train
from chemprop.data import StandardScaler
from chemprop.data.utils import get_class_sizes, get_data, get_task_names, split_data
from chemprop.models import build_model, build_pretrain_model, add_functional_prompt
from chemprop.models import build_tapt_model, add_tapt_prompt  # ✅ 从 model.py 导入
from chemprop.nn_utils import param_count
from chemprop.utils import build_optimizer, build_lr_scheduler, get_loss_func, get_metric_func, load_checkpoint, \
    makedirs, save_checkpoint
from chemprop.data import MoleculeDataset
from tqdm import tqdm, trange
from chemprop.models import ContrastiveLoss
from chemprop.torchlight import initialize_exp, snapshot
from torch.optim import Adam


def run_training(args: Namespace, prompt: bool, logger: Logger = None) -> List[float]:
    """
    Trains a model and returns test scores on the model checkpoint with the highest validation score.

    :param args: Arguments.
    :param prompt: Whether to use prompt-based training.
    :param logger: Logger.
    :return: A list of ensemble scores for each task.
    """
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Set GPU
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    # ============================================================
    # 数据加载与划分（只做一次，在循环外）
    # ============================================================
    info('Loading data')
    args.task_names = get_task_names(args.data_path)
    data = get_data(path=args.data_path, args=args, logger=logger)
    args.num_tasks = data.num_tasks()
    args.features_size = data.features_size()
    info(f'Number of tasks = {args.num_tasks}')

    # Split data（使用固定 seed）
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

    # Save SMILES splits (only once, not per run)
    if args.save_smiles_splits:
        with open(args.data_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)

            lines_by_smiles = {}
            indices_by_smiles = {}
            for i, line in enumerate(reader):
                smiles = line[0]
                lines_by_smiles[smiles] = line
                indices_by_smiles[smiles] = i

        all_split_indices = []
        for dataset, name in [(train_data, 'train'), (val_data, 'val'), (test_data, 'test')]:
            with open(os.path.join(args.save_dir, name + '_smiles.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['smiles'])
                for smiles in dataset.smiles():
                    writer.writerow([smiles])

            with open(os.path.join(args.save_dir, name + '_full.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for smiles in dataset.smiles():
                    writer.writerow(lines_by_smiles[smiles])

            split_indices = []
            for smiles in dataset.smiles():
                split_indices.append(indices_by_smiles[smiles])
            split_indices = sorted(split_indices)
            all_split_indices.append(split_indices)

        with open(os.path.join(args.save_dir, 'split_indices.pckl'), 'wb') as f:
            pickle.dump(all_split_indices, f)

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
    # 🚀 外层循环 - num_runs (相同数据，不同模型初始化)
    # ============================================================
    all_run_scores = []  # 收集所有 runs 的 ensemble 分数

    for run_idx in range(args.num_runs):
        info('\n' + '=' * 80)
        info(f'🚀 Starting Run {run_idx + 1}/{args.num_runs} (seed={args.seed}, data split is fixed)')
        info('=' * 80 + '\n')

        # 为当前 run 初始化预测累加器
        if args.dataset_type == 'multiclass':
            sum_test_preds = np.zeros((len(test_smiles), args.num_tasks, args.multiclass_num_classes))
        else:
            sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))

        # ============================================================
        # 内层循环 - Ensemble (相同 seed，不同初始化)
        # ============================================================
        for model_idx in range(args.ensemble_size):
            info(f'\n📦 Training Model {model_idx + 1}/{args.ensemble_size} (Run {run_idx + 1}/{args.num_runs})\n')

            # 为每个模型设置保存目录（区分 run 和 model）
            save_dir = os.path.join(args.save_dir, f'run_{run_idx}', f'model_{model_idx}')
            makedirs(save_dir)
            try:
                writer = SummaryWriter(log_dir=save_dir)
            except:
                writer = SummaryWriter(logdir=save_dir)

            # ============================================================
            # Load/build model (🚀 使用 model.py 中的 TAPT 实现)
            # ============================================================
            if getattr(args, 'use_tapt', False) or args.step == 'tapt':
                # ===== TAPT 模式 =====
                info('🚀 Building TAPT model using model.py implementation...')

                # 使用 model.py 中的官方 build_tapt_model
                model = build_tapt_model(
                    args=args,
                    encoder_name=args.encoder_name,
                    num_tasks=args.num_tasks
                )

                # 加载 KANO 预训练权重
                if args.checkpoint_path is not None:
                    info(f'Loading KANO checkpoint: {args.checkpoint_path}')
                    state_dict = torch.load(args.checkpoint_path, map_location='cpu')

                    # 加载 encoder
                    encoder_state = state_dict.get('encoder', state_dict)
                    missing_keys, unexpected_keys = model.encoder.load_state_dict(encoder_state, strict=False)

                    loaded_params = sum(p.numel() for p in model.encoder.parameters())
                    info(f'✅ KANO encoder weights loaded: {loaded_params:,} parameters')

                    if missing_keys:
                        debug(f'⚠️  Missing keys: {len(missing_keys)}')
                    if unexpected_keys:
                        debug(f'⚠️  Unexpected keys: {len(unexpected_keys)}')

                # 冻结 KANO 参数（如果指定）
                if getattr(args, 'freeze_kano', False):
                    info('🔒 Freezing KANO Encoder (keeping FFN trainable)...')

                    # 只冻结 encoder
                    for param in model.encoder.parameters():
                        param.requires_grad = False

                    # ✅ FFN 保持可训练（不冻结）
                    # ✅ TAPT 模块保持可训练（不冻结）

                    # 统计参数
                    kano_frozen = sum(p.numel() for p in model.encoder.parameters())

                    ffn_trainable = sum(p.numel() for p in model.ffn.parameters() if p.requires_grad) if hasattr(model,
                                                                                                                 'ffn') else 0

                    tapt_trainable = 0
                    if hasattr(model, 'tapt_module') and model.tapt_module is not None:
                        tapt_trainable += sum(p.numel() for p in model.tapt_module.parameters() if p.requires_grad)
                    if hasattr(model, 'prompt_injectors') and model.prompt_injectors is not None:
                        tapt_trainable += sum(p.numel() for p in model.prompt_injectors.parameters() if p.requires_grad)

                    total_trainable = ffn_trainable + tapt_trainable

                    info(f'❄️  KANO Encoder frozen: {kano_frozen:,} parameters')
                    info(f'✅ FFN trainable: {ffn_trainable:,} parameters')
                    info(f'✅ TAPT trainable: {tapt_trainable:,} parameters')
                    info(
                        f'📊 Total trainable: {total_trainable:,} parameters ({total_trainable / (kano_frozen + total_trainable) * 100:.2f}%)')

                    # 确保 TAPT 参数可训练
                    if hasattr(model, 'tapt_module') and model.tapt_module is not None:
                        for param in model.tapt_module.parameters():
                            param.requires_grad = True

                    # 统计
                    kano_frozen = sum(p.numel() for p in model.encoder.parameters())
                    if hasattr(model, 'ffn') and model.ffn is not None:
                        kano_frozen += sum(p.numel() for p in model.ffn.parameters())

                    tapt_trainable = sum(
                        p.numel() for p in model.tapt_module.parameters() if p.requires_grad) if hasattr(model,
                                                                                                         'tapt_module') else 0

                    info(f'❄️  KANO frozen: {kano_frozen:,} parameters')
                    info(f'✅ TAPT trainable: {tapt_trainable:,} parameters')
                    info(f'📊 Trainable ratio: {tapt_trainable / (kano_frozen + tapt_trainable) * 100:.2f}%')

                info(f'✅ TAPT model built with prompt_dim={getattr(args, "prompt_dim", 128)}')

            else:
                # ===== KANO 模式（原版） =====
                if args.checkpoint_path is not None:
                    debug(f'Loading model from {args.checkpoint_path}')
                    model = build_model(args, encoder_name=args.encoder_name)
                    model.encoder.load_state_dict(torch.load(args.checkpoint_path, map_location='cpu'), strict=False)
                else:
                    debug(f'Building model {model_idx}')
                    model = build_model(args, encoder_name=args.encoder_name)

                if args.step == 'functional_prompt':
                    add_functional_prompt(model, args)

            debug(model)
            debug(f'Number of parameters = {param_count(model):,}')

            if args.cuda:
                debug('Moving model to cuda')
                model = model.cuda()

            # Ensure that model is saved in correct location for evaluation if 0 epochs
            save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)

            # ============================================================
            # Optimizers (🚀 TAPT 差分学习率支持)
            # ============================================================
            if getattr(args, 'use_tapt', False) or args.step == 'tapt':
                if getattr(args, 'freeze_kano', False):
                    # ===== 只优化 TAPT 参数 =====
                    info('🎯 Optimizer mode: TAPT parameters only')

                    if not hasattr(model, 'tapt_module') or model.tapt_module is None:
                        raise AttributeError("❌ Model does not have 'tapt_module' attribute!")

                    # 收集 TAPT 参数
                    tapt_params = list(model.tapt_module.parameters())

                    # 添加 prompt_injectors 参数（如果存在）
                    if hasattr(model, 'prompt_injectors') and model.prompt_injectors is not None:
                        tapt_params.extend(list(model.prompt_injectors.parameters()))

                    # 验证参数列表不为空
                    if not tapt_params:
                        raise ValueError("❌ No TAPT parameters found! Check model structure.")

                    # 统计参数数量
                    total_tapt_params = sum(p.numel() for p in tapt_params)
                    trainable_tapt_params = sum(p.numel() for p in tapt_params if p.requires_grad)

                    info(f'📦 TAPT parameters: {total_tapt_params:,} (trainable: {trainable_tapt_params:,})')

                    if trainable_tapt_params == 0:
                        raise RuntimeError("❌ No trainable TAPT parameters found!")

                    # 创建 optimizer
                    optimizer = torch.optim.Adam(
                        tapt_params,
                        lr=getattr(args, 'prompt_lr', 1e-4),
                        weight_decay=getattr(args, 'weight_decay', 0.0)
                    )
                    info(f'✅ Optimizer: Adam (TAPT only, lr={getattr(args, "prompt_lr", 1e-4)})')

                else:
                    # ===== 差分学习率（KANO + TAPT 都训练） =====
                    info('🎯 Optimizer mode: Differential learning rates (KANO + TAPT)')

                    # 分组参数
                    kano_params = []
                    tapt_params = []

                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            if 'tapt' in name.lower() or 'prompt' in name.lower():
                                tapt_params.append(param)
                            else:
                                kano_params.append(param)

                    param_groups = [
                        {'params': kano_params, 'lr': getattr(args, 'kano_lr', 1e-4)},
                        {'params': tapt_params, 'lr': getattr(args, 'prompt_lr', 1e-4)}
                    ]

                    optimizer = torch.optim.Adam(
                        param_groups,
                        weight_decay=getattr(args, 'weight_decay', 0.0)
                    )
                    info(
                        f'✅ Optimizer: Adam (KANO lr={getattr(args, "kano_lr", 1e-4)}, TAPT lr={getattr(args, "prompt_lr", 1e-4)})')
            else:
                # ===== 原版 KANO Optimizer =====
                optimizer = build_optimizer(model, args)

            # ============================================================
            # Learning rate schedulers (🚀 适配 freeze_kano)
            # ============================================================
            if getattr(args, 'use_tapt', False) or args.step == 'tapt':
                if getattr(args, 'freeze_kano', False):
                    # ===== Freeze 模式：使用简单的 ExponentialLR =====
                    info('📊 Scheduler: ExponentialLR (TAPT only, gamma=0.99)')
                    scheduler = ExponentialLR(optimizer, gamma=0.99)
                else:
                    # ===== 非 Freeze 模式：使用原版差分学习率 scheduler =====
                    info('📊 Scheduler: NoamLR (differential LR for KANO + TAPT)')
                    scheduler = build_lr_scheduler(optimizer, args)
            else:
                # ===== 原版 KANO Scheduler =====
                scheduler = build_lr_scheduler(optimizer, args)

            # ============================================================
            # 📊 打印模型架构信息（只在第一个 run、第一个 model 时）
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

                # 只在 TAPT 模式下统计 TAPT 参数
                if (getattr(args, 'use_tapt', False) or args.step == 'tapt') and hasattr(model,
                                                                                         'tapt_module') and model.tapt_module is not None:
                    tapt_params = sum(p.numel() for p in model.tapt_module.parameters() if p.requires_grad)
                    info(f'TAPT parameters: {tapt_params:,} ({tapt_params / trainable_params * 100:.2f}% of trainable)')

                info('=' * 80 + '\n')

            # ============================================================
            # Run training (Added Early Stopping)
            # ============================================================
            best_score = float('inf') if args.minimize_score else -float('inf')
            best_epoch, n_iter = 0, 0

            # ✅ 新增：初始化耐心值计数器
            patience = 0
            patience_limit = getattr(args, 'patience', 30)  # 默认 30 epoch 没提升就停

            for epoch in range(args.epochs):
                info(f'Epoch {epoch}')

                n_iter = train(
                    model=model,
                    prompt=prompt,
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

                val_scores = evaluate(
                    model=model,
                    prompt=prompt,
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

                test_preds = predict(
                    model=model,
                    prompt=prompt,
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
                info(f'test {args.metric} = {avg_test_score:.6f}')

                if args.show_individual_scores:
                    for task_name, val_score in zip(args.task_names, val_scores):
                        debug(f'Validation {task_name} {args.metric} = {val_score:.6f}')
                        writer.add_scalar(f'validation_{task_name}_{args.metric}', val_score, n_iter)

                # ============================================================
                # ✅ 修改：加入早停逻辑 (Early Stopping)
                # ============================================================
                if args.minimize_score and avg_val_score < best_score or \
                        not args.minimize_score and avg_val_score > best_score:
                    best_score, best_epoch = avg_val_score, epoch
                    save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)
                    # 🚀 只要有提升，就重置耐心值
                    patience = 0
                else:
                    # 🚀 没有提升，耐心值 +1
                    patience += 1
                    debug(f'⚠️ No improvement. Patience: {patience}/{patience_limit}')

                # 🚀 触发早停
                if patience >= patience_limit:
                    info(f'🛑 Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch}.')
                    break

            # Evaluate on test set using model with best validation score
            info(f'Model {model_idx} best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')

            original_log_level = None
            if logger:
                original_log_level = logger.level
                logger.setLevel(logging.INFO)

            model = load_checkpoint(os.path.join(save_dir, 'model.pt'), cuda=args.cuda, logger=logger)

            if logger and original_log_level:
                logger.setLevel(original_log_level)

            test_preds = predict(
                model=model,
                prompt=prompt,
                data=test_data,
                batch_size=args.batch_size,
                scaler=scaler
            )

            test_preds = predict(
                model=model,
                prompt=prompt,
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

            if args.show_individual_scores:
                for task_name, test_score in zip(args.task_names, test_scores):
                    info(f'Model {model_idx} test {task_name} {args.metric} = {test_score:.6f}')
                    writer.add_scalar(f'test_{task_name}_{args.metric}', test_score, n_iter)

        # ============================================================
        # 评估当前 run 的 Ensemble 性能
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

        # 收集当前 run 的分数
        all_run_scores.append(avg_ensemble_test_score)

        if args.show_individual_scores:
            for task_name, ensemble_score in zip(args.task_names, ensemble_scores):
                info(f'Ensemble test {task_name} {args.metric} = {ensemble_score:.6f}')

    # ============================================================
    # 计算并报告所有 runs 的统计结果
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

        return all_run_scores
    else:
        # 只有一次 run，直接返回 ensemble 分数
        info(f'\n✅ Final Test {args.metric} = {all_run_scores[0]:.6f}')
        return [all_run_scores[0]]


def pre_training(args: Namespace, logger: Logger = None) -> List[float]:
    """
    Trains a model and returns test scores on the model checkpoint with the highest validation score.

    :param args: Arguments.
    :param logger: Logger.
    :return: A list of ensemble scores for each task.
    """
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Set GPU
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    # Get data
    debug('Loading data')
    data = get_data(path=args.data_path, args=args, logger=logger)

    args.data_size = len(data)

    debug(f'Total size = {len(data)}')

    # Train ensemble of models
    for model_idx in range(args.ensemble_size):
        # Tensorboard writer
        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        makedirs(save_dir)

        # Load/build model
        if args.checkpoint_paths is not None:
            debug(f'Loading model {model_idx} from {args.checkpoint_paths[model_idx]}')
            model = load_checkpoint(args.checkpoint_paths[model_idx], current_args=args, logger=logger)
        else:
            debug(f'Building model {model_idx}')
            model1 = build_pretrain_model(args, encoder_name='CMPNN')
            model2 = build_pretrain_model(args, encoder_name='CMPNN')

        debug(model1)
        debug(f'Number of M1 parameters = {param_count(model1):,}')
        if args.cuda:
            debug('Moving model to cuda')
            model1 = model1.cuda()

        debug(model2)
        debug(f'Number of M2 parameters = {param_count(model2):,}')
        if args.cuda:
            debug('Moving model to cuda')
            model2 = model2.cuda()

        logger, dump_folder = initialize_exp(Namespace(**args.__dict__))
        dump_folder = f'{dump_folder}-model'

        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        args.device = device
        criterion = ContrastiveLoss(loss_computer='nce_softmax', temperature=args.temperature, args=args).cuda()
        optimizer = Adam([{"params": model1.parameters()}, {"params": model2.parameters()}], lr=3e-5)
        scheduler = ExponentialLR(optimizer, 0.99, -1)
        step_per_schedule = 500
        global_step = 0

        mol = MoleculeDataset(data)
        smiles, features = mol.smiles(), mol.features()

        loader = DataLoader(smiles,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=12,
                            drop_last=True)

        # Run training
        for epoch in range(args.epochs):
            debug(f'Epoch {epoch}')

            debug = logger.debug if logger is not None else print
            model1.train()
            model2.train()

            step = 'pretrain'
            for batch in tqdm(loader):
                # Run model
                emb1 = model1(step, False, batch, None)
                emb2 = model2(step, True, batch, None)

                loss = criterion(emb1, emb2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                global_step += 1

                # save model
                if global_step % 1000 == 0:
                    snapshot(model1.encoder, global_step, dump_folder, 'original')
                    snapshot(model2.encoder, global_step, dump_folder, 'augment')
                if global_step % step_per_schedule == 0:
                    scheduler.step()
            logger.info(f'[{epoch}/{args.epochs}] train loss {loss.item():.4f}')

