import random
import sys
import logging
from chemprop.parsing import parse_train_args
from chemprop.train import run_training


def setup_logging(args):
    logger = logging.getLogger('TAPT')
    logger.setLevel(logging.DEBUG if not args.quiet else logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if not args.quiet else logging.INFO)

    import os
    log_dir = f'./logs/{args.exp_name}/{args.exp_id}'
    os.makedirs(log_dir, exist_ok=True)
    log_file = f'{log_dir}/training.log'

    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.DEBUG)

    if args.use_tapt:
        formatter = logging.Formatter(
            '%(asctime)s [TAPT] %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s [BASE] %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info(f'Logging to file: {log_file}')

    return logger


def print_experiment_info(args, logger):
    logger.info('=' * 80)
    logger.info('EXPERIMENT CONFIGURATION')
    logger.info('=' * 80)

    logger.info(f'Experiment Name: {args.exp_name}')
    logger.info(f'Experiment ID: {args.exp_id}')
    logger.info(f'Mode: {"TAPT" if args.use_tapt else "Baseline（Functional Prompt）"}')
    logger.info(f'Step: {args.step}')

    logger.info('-' * 80)
    logger.info('Dataset Configuration:')
    logger.info(f'  Data Path: {args.data_path}')
    logger.info(f'  Dataset Type: {args.dataset_type}')
    logger.info(f'  Split Type: {args.split_type}')
    logger.info(f'  Split Sizes: {args.split_sizes}')
    logger.info(f'  Metric: {args.metric}')

    logger.info('-' * 80)
    logger.info('Training Configuration:')
    logger.info(f'  Epochs: {args.epochs}')
    logger.info(f'  Batch Size: {args.batch_size}')
    logger.info(f'  Num Runs: {args.num_runs}')
    logger.info(f'  Seed: {args.seed}')
    logger.info(f'  GPU: {args.gpu if args.gpu is not None else "CPU"}')

    logger.info('-' * 80)
    logger.info('Model Configuration:')
    logger.info(f'  Encoder: {args.encoder_name}')
    logger.info(f'  Hidden Size: {args.hidden_size}')
    logger.info(f'  Depth: {args.depth}')
    logger.info(f'  Dropout: {args.dropout}')
    logger.info(f'  FFN Hidden Size: {args.ffn_hidden_size}')
    logger.info(f'  FFN Num Layers: {args.ffn_num_layers}')

    if args.use_tapt:
        logger.info('-' * 80)
        logger.info('TAPT Configuration:')
        logger.info(f'  Prompt Dimension: {args.prompt_dim}')
        logger.info(f'  Prompt Tokens: {args.num_prompt_tokens}')
        logger.info(f'  Task ID: {args.task_id}')
        logger.info(f'  Backbone Learning Rate: {args.backbone_lr}')
        logger.info(f'  Prompt Learning Rate: {args.prompt_lr}')
        logger.info(f'  Freeze Encoder: {args.freeze_encoder}')
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

    if args.checkpoint_path:
        logger.info('-' * 80)
        logger.info('Checkpoint Configuration:')
        logger.info(f'  Checkpoint Path: {args.checkpoint_path}')

    logger.info('=' * 80)
    logger.info('')


def validate_tapt_requirements(args, logger):
    if not args.use_tapt:
        return True

    logger.info('Validating TAPT requirements...')

    try:
        from chemprop.models import (
            build_tapt_model,
        )
        logger.info('TAPT model module found')
    except ImportError as e:
        logger.error('TAPT model module not found!')
        logger.error(f'Error: {e}')
        logger.error('Please ensure chemprop/models/tapt_modules.py exists.')
        return False

    if args.checkpoint_path is None and args.checkpoint_dir is None:
        logger.error('TAPT requires a pre-trained checkpoint!')
        logger.error('Please specify --checkpoint_path or --checkpoint_dir.')
        return False

    logger.info('All TAPT requirements satisfied')
    return True


def main():
    try:
        args = parse_train_args()

        if args.seed == -1:
            args.seed = random.randint(1, 10000)

        logger = setup_logging(args)

        if not args.quiet:
            print_experiment_info(args, logger)

        if not validate_tapt_requirements(args, logger):
            logger.error('Validation failed. Exiting...')
            sys.exit(1)

        logger.info('Starting training...')
        logger.info('')

        prompt = (args.step == 'functional_prompt')

        run_training(args, prompt, logger)

        logger.info('')
        logger.info('=' * 80)
        logger.info('Training completed successfully!')
        logger.info('=' * 80)

    except KeyboardInterrupt:
        print('\n')
        print('=' * 80)
        print('Training interrupted by user')
        print('=' * 80)
        sys.exit(0)

    except Exception as e:
        print('\n')
        print('=' * 80)
        print('Training failed with error:')
        print(f'   {type(e).__name__}: {e}')
        print('=' * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
