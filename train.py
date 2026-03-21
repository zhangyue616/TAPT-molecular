# ===== train.py (TAPT 集成版) =====
"""
KANO/TAPT 训练脚本

支持两种模式：
1. KANO 模式 (原版): python train.py --step functional_prompt ...
2. TAPT 模式 (新增): python train.py --use_tapt --step tapt ...

作者: Modified for TAPT integration
日期: 2026-01-19
"""
import random
import sys
import logging
from chemprop.parsing import parse_train_args
from chemprop.train import run_training


def setup_logging(args):
    """设置日志系统"""
    # 创建日志记录器
    logger = logging.getLogger('KANO-TAPT')
    logger.setLevel(logging.DEBUG if not args.quiet else logging.INFO)

    # ✅ 创建控制台处理器（保留原有）
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if not args.quiet else logging.INFO)

    # 🆕 创建文件处理器（新增）
    import os
    log_dir = f'./logs/{args.exp_name}/{args.exp_id}'
    os.makedirs(log_dir, exist_ok=True)
    log_file = f'{log_dir}/training.log'

    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.DEBUG)

    # 设置日志格式
    if args.use_tapt:
        formatter = logging.Formatter(
            '%(asctime)s [TAPT] %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s [KANO] %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)  # 🆕 文件也用相同格式

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)  # 🆕 添加文件处理器

    # 🆕 记录日志文件位置
    logger.info(f'Logging to file: {log_file}')

    return logger


def print_experiment_info(args, logger):
    """
    打印实验配置信息

    Args:
        args: 参数命名空间
        logger: 日志记录器
    """
    logger.info('=' * 80)
    logger.info('EXPERIMENT CONFIGURATION')
    logger.info('=' * 80)

    # 基本信息
    logger.info(f'Experiment Name: {args.exp_name}')
    logger.info(f'Experiment ID: {args.exp_id}')
    logger.info(f'Mode: {"TAPT" if args.use_tapt else "KANO"}')
    logger.info(f'Step: {args.step}')

    # 数据集信息
    logger.info('-' * 80)
    logger.info('Dataset Configuration:')
    logger.info(f'  Data Path: {args.data_path}')
    logger.info(f'  Dataset Type: {args.dataset_type}')
    logger.info(f'  Split Type: {args.split_type}')
    logger.info(f'  Split Sizes: {args.split_sizes}')
    logger.info(f'  Metric: {args.metric}')

    # 训练配置
    logger.info('-' * 80)
    logger.info('Training Configuration:')
    logger.info(f'  Epochs: {args.epochs}')
    logger.info(f'  Batch Size: {args.batch_size}')
    logger.info(f'  Num Runs: {args.num_runs}')
    logger.info(f'  Seed: {args.seed}')
    logger.info(f'  GPU: {args.gpu if args.gpu is not None else "CPU"}')

    # 模型配置
    logger.info('-' * 80)
    logger.info('Model Configuration:')
    logger.info(f'  Encoder: {args.encoder_name}')
    logger.info(f'  Hidden Size: {args.hidden_size}')
    logger.info(f'  Depth: {args.depth}')
    logger.info(f'  Dropout: {args.dropout}')
    logger.info(f'  FFN Hidden Size: {args.ffn_hidden_size}')
    logger.info(f'  FFN Num Layers: {args.ffn_num_layers}')

    # TAPT 特定配置
    if args.use_tapt:
        logger.info('-' * 80)
        logger.info('🚀 TAPT Configuration:')
        logger.info(f'  Prompt Dimension: {args.prompt_dim}')
        logger.info(f'  Prompt Tokens: {args.num_prompt_tokens}')
        logger.info(f'  Task ID: {args.task_id}')
        logger.info(f'  KANO Learning Rate: {args.kano_lr}')
        logger.info(f'  Prompt Learning Rate: {args.prompt_lr}')
        logger.info(f'  Freeze KANO: {args.freeze_kano}')
        logger.info(f'  Injection Layers: {args.prompt_injection_layers}')
        logger.info(f'  TAPT Dropout: {args.tapt_dropout}')
        logger.info(f'  Weight Decay: {args.weight_decay}')
    else:
        logger.info('-' * 80)
        logger.info('Learning Rate Configuration:')
        logger.info(f'  Initial LR: {args.init_lr}')
        logger.info(f'  Max LR: {args.max_lr}')
        logger.info(f'  Final LR: {args.final_lr}')
        logger.info(f'  Warmup Epochs: {args.warmup_epochs}')

    # 检查点信息
    if args.checkpoint_path:
        logger.info('-' * 80)
        logger.info('Checkpoint Configuration:')
        logger.info(f'  Checkpoint Path: {args.checkpoint_path}')

    logger.info('=' * 80)
    logger.info('')


def validate_tapt_requirements(args, logger):
    """
    验证 TAPT 模式的必要条件

    Args:
        args: 参数命名空间
        logger: 日志记录器
    Returns:
        bool: 验证是否通过
    """
    if not args.use_tapt:
        return True

    logger.info('Validating TAPT requirements...')

    # ✅ 修正导入检查：应该检查 model.py 中的函数
    try:
        # 从 chemprop.models (即 model.py) 导入
        from chemprop.models import build_tapt_model
        logger.info('✅ TAPT model module found')
    except ImportError as e:
        logger.error('❌ TAPT functions not found in chemprop.models!')
        logger.error(f'Error: {e}')
        return False

    # 检查预训练权重
    if args.checkpoint_path is None and args.checkpoint_dir is None:
        logger.error('❌ TAPT requires a pre-trained checkpoint!')
        logger.error('Please specify --checkpoint_path or --checkpoint_dir.')
        return False

    logger.info('✅ All TAPT requirements satisfied')
    return True


def main():
    """
    主函数
    """
    try:
        # 解析参数
        args = parse_train_args()

        # ==========================================
        # 🎲【新增】随机种子自动生成逻辑
        # ==========================================
        # 如果输入 --seed -1，则开启“随机模式”
        if args.seed == -1:
            # 方案 A: 完全随机 (1 到 10000 之间)
            random_seed = random.randint(1, 10000)

            # 方案 B (可选): 从您的“幸运数字”列表中随机选
            # lucky_seeds = [4, 42, 88, 123, 2024, 2026]
            # random_seed = random.choice(lucky_seeds)

            # 覆盖 args 中的 seed
            args.seed = random_seed
            print(f"\n🎲 [Auto-Seed] Mode Enabled! Randomly selected seed: {args.seed}\n")
        # ==========================================

        # 设置日志 (此时 args.seed 已经被更新为随机数了)
        logger = setup_logging(args)

        # 打印实验配置
        if not args.quiet:
            print_experiment_info(args, logger)

        # 验证 TAPT 要求
        if not validate_tapt_requirements(args, logger):
            logger.error('Validation failed. Exiting...')
            sys.exit(1)

        # 运行训练
        logger.info('Starting training...')
        logger.info('')

        prompt = (args.step == 'functional_prompt')

        run_training(args, prompt, logger)

        logger.info('')
        logger.info('=' * 80)
        logger.info('✅ Training completed successfully!')
        logger.info('=' * 80)

    except KeyboardInterrupt:
        print('\n')
        print('=' * 80)
        print('⚠️  Training interrupted by user')
        print('=' * 80)
        sys.exit(0)

    except Exception as e:
        print('\n')
        print('=' * 80)
        print('❌ Training failed with error:')
        print(f'   {type(e).__name__}: {e}')
        print('=' * 80)

        # 打印详细错误信息（调试模式）
        import traceback
        traceback.print_exc()

        sys.exit(1)


if __name__ == '__main__':
    main()
