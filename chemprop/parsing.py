# ===== chemprop/parsing.py (TAPT 集成完整版 + 结构噪声控制) =====
from argparse import ArgumentParser, Namespace
import json
import os
import pickle

import torch

from chemprop.utils import makedirs
from chemprop.features import get_available_features_generators


def add_predict_args(parser: ArgumentParser):
    """
    Adds predict arguments to an ArgumentParser.

    :param parser: An ArgumentParser.
    """
    parser.add_argument('--gpu', type=int,
                        choices=list(range(torch.cuda.device_count())),
                        help='Which GPU to use')
    parser.add_argument('--test_path', type=str,
                        help='Path to CSV file containing testing data for which predictions will be made',
                        default='../input/test.csv')
    parser.add_argument('--use_compound_names', action='store_true', default=False,
                        help='Use when test data file contains compound names in addition to SMILES strings')
    parser.add_argument('--preds_path', type=str,
                        help='Path to CSV file where predictions will be saved',
                        default='test_pred')
    parser.add_argument('--checkpoint_dir', type=str,
                        help='Directory from which to load model checkpoints'
                             '(walks directory and ensembles all models that are found)',
                        default='./ckpt')
    parser.add_argument('--checkpoint_path', type=str,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Turn off cuda')
    parser.add_argument('--features_generator', type=str, nargs='*',
                        choices=get_available_features_generators(),
                        help='Method of generating additional features')
    parser.add_argument('--features_path', type=str, nargs='*',
                        help='Path to features to use in FNN (instead of features_generator)')
    parser.add_argument('--no_features_scaling', action='store_true', default=False,
                        help='Turn off scaling of features')
    parser.add_argument('--max_data_size', type=int,
                        help='Maximum number of data points to load')


def add_train_args(parser: ArgumentParser):
    """
    Adds training arguments to an ArgumentParser.

    :param parser: An ArgumentParser.
    """
    # ============================================================
    # General arguments
    # ============================================================
    parser.add_argument('--gpu', type=int,
                        choices=list(range(torch.cuda.device_count())),
                        help='Which GPU to use')
    parser.add_argument('--data_path', type=str,
                        help='Path to data CSV file',
                        default='M_CYP1A2I_I.csv')
    parser.add_argument('--mode', type=str, default='mode1',
                        choices=['mode1', 'pure_kano',
                                 'mode2', 'task_only',
                                 'mode3', 'kano_task',
                                 'mode4', 'kano_kg_task'],
                        help='TAPT Operation Mode:\n'
                             '  mode1: Pure KANO (Baseline)\n'
                             '  mode2: Task-Only (No KG)\n'
                             '  mode3: KANO + Task (Hybrid)\n'
                             '  mode4: KANO + KG + Task (Full Synergy)')
    parser.add_argument('--use_compound_names', action='store_true', default=False,
                        help='Use when test data file contains compound names in addition to SMILES strings')
    parser.add_argument('--max_data_size', type=int,
                        help='Maximum number of data points to load')
    parser.add_argument('--test', action='store_true', default=False,
                        help='Whether to skip training and only test the model')
    parser.add_argument('--features_only', action='store_true', default=False,
                        help='Use only the additional features in an FFN, no graph network')
    parser.add_argument('--features_generator', type=str, nargs='*',
                        choices=get_available_features_generators(),
                        help='Method of generating additional features')
    parser.add_argument('--features_path', type=str, nargs='*',
                        help='Path to features to use in FNN (instead of features_generator)')
    parser.add_argument('--save_dir', type=str, default='./ckpt',
                        help='Directory where model checkpoints will be saved')
    parser.add_argument('--save_smiles_splits', action='store_true', default=False,
                        help='Save smiles for each train/val/test splits for prediction convenience later')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory from which to load model checkpoints'
                             '(walks directory and ensembles all models that are found)')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--dataset_type', type=str,
                        choices=['classification', 'regression', 'multiclass'],
                        help='Type of dataset, e.g. classification or regression.'
                             'This determines the loss function used during training.',
                        default='regression')
    parser.add_argument('--multiclass_num_classes', type=int, default=3,
                        help='Number of classes when running multiclass classification')
    parser.add_argument('--separate_val_path', type=str,
                        help='Path to separate val set, optional')
    parser.add_argument('--separate_val_features_path', type=str, nargs='*',
                        help='Path to file with features for separate val set')
    parser.add_argument('--separate_test_path', type=str,
                        help='Path to separate test set, optional')
    parser.add_argument('--separate_test_features_path', type=str, nargs='*',
                        help='Path to file with features for separate test set')
    parser.add_argument('--split_type', type=str, default='random',
                        choices=['random', 'scaffold_balanced', 'predetermined', 'crossval', 'index_predetermined',
                                 'cluster_balanced'],
                        help='Method of splitting the data into train/val/test')
    parser.add_argument('--split_sizes', type=float, nargs=3, default=[0.8, 0.1, 0.1],
                        help='Split proportions for train/validation/test sets')
    parser.add_argument('--num_runs', type=int, default=1,
                        help='Number of runs when performing k independent runs')
    parser.add_argument('--folds_file', type=str, default=None,
                        help='Optional file of fold labels')
    parser.add_argument('--val_fold_index', type=int, default=None,
                        help='Which fold to use as val for leave-one-out cross val')
    parser.add_argument('--test_fold_index', type=int, default=None,
                        help='Which fold to use as test for leave-one-out cross val')
    parser.add_argument('--crossval_index_dir', type=str,
                        help='Directory in which to find cross validation index files')
    parser.add_argument('--crossval_index_file', type=str,
                        help='Indices of files to use as train/val/test'
                             'Overrides --num_folds and --seed.')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed to use when splitting data into train/val/test sets.'
                             'When `num_runs` > 1, the first run uses this seed and all'
                             'subsequent runs add 1 to the seed.')
    parser.add_argument('--metric', type=str, default=None,
                        choices=['auc', 'prc-auc', 'rmse', 'mae', 'mse', 'r2', 'accuracy', 'cross_entropy'],
                        help='Metric to use during evaluation.'
                             'Note: Does NOT affect loss function used during training'
                             '(loss is determined by the `dataset_type` argument).'
                             'Note: Defaults to "auc" for classification and "rmse" for regression.')
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='Skip non-essential print statements')
    parser.add_argument('--log_frequency', type=int, default=10,
                        help='The number of batches between each logging of the training loss')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Turn off cuda')
    parser.add_argument('--show_individual_scores', action='store_true', default=False,
                        help='Show all scores for individual targets, not just average, at the end')
    parser.add_argument('--no_cache', action='store_true', default=False,
                        help='Turn off caching mol2graph computation')
    parser.add_argument('--config_path', type=str,
                        help='Path to a .json file containing arguments. Any arguments present in the config'
                             'file will override arguments specified via the command line or by the defaults.')

    # ============================================================
    # Training arguments
    # ============================================================
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size')
    parser.add_argument('--warmup_epochs', type=float, default=2.0,
                        help='Number of epochs during which learning rate increases linearly from'
                             'init_lr to max_lr. Afterwards, learning rate decreases exponentially'
                             'from max_lr to final_lr.')
    parser.add_argument('--init_lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--max_lr', type=float, default=1e-3,
                        help='Maximum learning rate')
    parser.add_argument('--final_lr', type=float, default=1e-4,
                        help='Final learning rate')
    parser.add_argument('--no_features_scaling', action='store_true', default=False,
                        help='Turn off scaling of features')

    # ============================================================
    # Model arguments
    # ============================================================
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Temperature of contrastive learning')
    parser.add_argument('--encoder_name', type=str, default='CMPNN',
                        choices=['CMPNN', 'MPNN'],
                        help='Name of the encoder')
    parser.add_argument('--ensemble_size', type=int, default=1,
                        help='Number of models in ensemble')
    parser.add_argument('--hidden_size', type=int, default=300,
                        help='Dimensionality of hidden layers in MPN')
    parser.add_argument('--bias', action='store_true', default=False,
                        help='Whether to add bias to linear layers')
    parser.add_argument('--depth', type=int, default=3,
                        help='Number of message passing steps')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout probability')
    parser.add_argument('--activation', type=str, default='ReLU',
                        choices=['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU', 'GELU'],
                        help='Activation function')
    parser.add_argument('--undirected', action='store_true', default=False,
                        help='Undirected edges (always sum the two relevant bond vectors)')
    parser.add_argument('--ffn_hidden_size', type=int, default=None,
                        help='Hidden dim for higher-capacity FFN (defaults to hidden_size)')
    parser.add_argument('--ffn_num_layers', type=int, default=2,
                        help='Number of layers in FFN after MPN encoding')
    parser.add_argument('--atom_messages', action='store_true', default=False,
                        help='Use messages on atoms instead of messages on bonds')

    # ============================================================
    # Experiment arguments
    # ============================================================
    parser.add_argument("--dump_path", default="dumped", type=str,
                        help="Experiment dump path")
    parser.add_argument("--exp_name", default="", type=str, required=True,
                        help="Experiment name")
    parser.add_argument("--exp_id", default="", type=str, required=True,
                        help="Experiment ID")
    parser.add_argument("--step", default="functional_prompt", type=str,
                        choices=['pretrain', 'functional_prompt', 'finetune_add', 'finetune_concat', 'tapt'],
                        help="Training mode: pretrain, functional_prompt (KANO), or tapt (TAPT)")

    # ============================================================
    # TAPT-specific arguments
    # ============================================================
    parser.add_argument('--use_tapt', action='store_true', default=False,
                        help='Enable TAPT mode')

    # 🆕 新增：纯任务复现模式开关
    parser.add_argument('--only_task', action='store_true', default=False,
                        help='Enable Task-Only Reproduction Mode. '
                             'When enabled, KANO Functional Prompt is disabled, '
                             'and TAPT uses only task prompts with stochastic noise. '
                             'Use this flag to reproduce previous BBBP/ToxCast results.')

    # ============================================================
    # 🆕 结构特征控制开关（核心修改）
    # ============================================================
    parser.add_argument('--enable_structure_prompt', action='store_true', default=False,
                        help='Enable真实结构特征提取（默认False使用高性能随机噪声增强）。'
                             '不指定此参数时，模型使用经过验证的噪声正则化机制（复现高性能）；'
                             '指定此参数后，模型将尝试提取真实分子结构特征（实验性功能）。'
                             '⚠️ 在 --only_task 模式下此参数会被强制忽略。')

    # ============================================================
    # TAPT Full (4-Module) arguments
    # ============================================================
    parser.add_argument('--use_tapt_full', action='store_true', default=False,
                        help='Enable full 4-module TAPT (Task + Struct + Pyramid + Node)')

    # 4 个模块的开关
    parser.add_argument('--enable_task_prompt', action='store_true', default=True,
                        help='Enable Task Prompt Pool module')
    parser.add_argument('--enable_struct_prompt', action='store_true', default=True,
                        help='Enable Structure-Aware Prompt Generator module')
    parser.add_argument('--enable_pyramid_agg', action='store_true', default=True,
                        help='Enable Hierarchical Pyramid Aggregator module')
    parser.add_argument('--enable_node_injection', action='store_true', default=True,
                        help='Enable Node-Level Prompt Refiner module')

    # ============================================================
    # TAPT minimal perturbation control
    # ============================================================
    parser.add_argument('--tapt_alpha', type=float, default=0.001,
                        help='TAPT prompt mixing weight (default: 0.001, almost no effect)')

    # 冻结特定模块（用于 Ablation）
    parser.add_argument('--freeze_task_prompt', action='store_true', default=False,
                        help='Freeze Task Prompt module (ablation)')
    parser.add_argument('--freeze_struct_prompt', action='store_true', default=False,
                        help='Freeze Struct Prompt module (ablation)')
    parser.add_argument('--freeze_pyramid_agg', action='store_true', default=False,
                        help='Freeze Pyramid Aggregator module (ablation)')
    parser.add_argument('--freeze_node_injection', action='store_true', default=False,
                        help='Freeze Node Injection module (ablation)')
    parser.add_argument('--structure_noise_scale',
                        type=float,
                        default=0.01,
                        help='结构噪声系数 (0=纯净, >0=噪声增强)')
    # TAPT Full 特定参数
    parser.add_argument('--task_pool_size', type=int, default=3,
                        help='Size of task prompt pool (default: 3)')
    parser.add_argument('--num_struct_patterns', type=int, default=5,
                        help='Number of structural patterns in Struct Prompt Generator (default: 5)')
    parser.add_argument('--tapt_lr', type=float, default=1e-5,
                        help='Learning rate for TAPT modules (default: 1e-5)')

    parser.add_argument('--prompt_dim', type=int, default=128,
                        help='Dimension of TAPT prompt vectors (default: 128)')

    parser.add_argument('--num_prompt_tokens', type=int, default=5,
                        help='Number of learnable prompt tokens per task (default: 5)')

    parser.add_argument('--kano_lr', type=float, default=1e-5,
                        help='Learning rate for KANO encoder (default: 1e-5)')

    parser.add_argument('--prompt_lr', type=float, default=1e-3,
                        help='Learning rate for TAPT prompt modules (default: 1e-3)')

    parser.add_argument('--task_id', type=int, default=0,
                        help='Task ID for TAPT (default: 0 for single-task)')

    parser.add_argument('--freeze_kano', action='store_true', default=False,
                        help='Freeze KANO parameters, only train TAPT prompts')

    parser.add_argument('--prompt_injection_layers', type=int, nargs='+', default=None,
                        help='GNN layers to inject prompts (e.g., 0 1 2). If None, use all layers.')

    parser.add_argument('--tapt_dropout', type=float, default=0.1,
                        help='Dropout rate for TAPT components (default: 0.1)')

    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='L2 regularization weight decay (default: 0.0)')



def update_checkpoint_args(args: Namespace):
    """
    Walks the checkpoint directory to find all checkpoints, updating args.checkpoint_paths and args.ensemble_size.

    :param args: Arguments.
    """
    if hasattr(args, 'checkpoint_paths') and args.checkpoint_paths is not None:
        return

    if args.checkpoint_dir is not None and args.checkpoint_path is not None:
        raise ValueError('Only one of checkpoint_dir and checkpoint_path can be specified.')

    if args.checkpoint_dir is None:
        args.checkpoint_paths = [args.checkpoint_path] if args.checkpoint_path is not None else None
        return

    args.checkpoint_paths = []

    for root, _, files in os.walk(args.checkpoint_dir):
        for fname in files:
            if fname.endswith('.pt'):
                args.checkpoint_paths.append(os.path.join(root, fname))

    args.ensemble_size = len(args.checkpoint_paths)

    if args.ensemble_size == 0:
        raise ValueError(f'Failed to find any model checkpoints in directory "{args.checkpoint_dir}"')


def modify_predict_args(args: Namespace):
    """
    Modifies and validates predicting args in place.

    :param args: Arguments.
    """
    assert args.test_path
    assert args.preds_path
    assert args.checkpoint_dir is not None or args.checkpoint_path is not None or args.checkpoint_paths is not None

    update_checkpoint_args(args)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    del args.no_cuda

    # Create directory for preds path
    makedirs(args.preds_path, isfile=True)


def parse_predict_args() -> Namespace:
    parser = ArgumentParser()
    add_predict_args(parser)
    args = parser.parse_args()
    modify_predict_args(args)

    return args


def modify_train_args(args: Namespace):
    """
    Modifies and validates training arguments in place.

    :param args: Arguments.
    """
    global temp_dir

    # Load config file
    if args.config_path is not None:
        with open(args.config_path) as f:
            config = json.load(f)
            for key, value in config.items():
                setattr(args, key, value)

    # ============================================================
    # ✅ 实验路径管理：根据 exp_name 和 exp_id 设置 save_dir
    # ============================================================
    if hasattr(args, 'exp_name') and hasattr(args, 'exp_id') and args.exp_name and args.exp_id:
        # 如果 save_dir 是默认值（./ckpt），则自动生成基于实验名称的路径
        if args.save_dir == './ckpt':
            args.save_dir = os.path.join(args.dump_path, args.exp_name, args.exp_id)
            if not args.quiet:
                print(f'✅ Auto-setting save_dir = {args.save_dir}')

    # 创建保存目录
    makedirs(args.save_dir)

    assert args.data_path is not None
    assert args.dataset_type is not None

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    del args.no_cuda

    args.features_scaling = not args.no_features_scaling
    del args.no_features_scaling

    # ============================================================
    # Metric 默认值设置
    # ============================================================
    if args.metric is None:
        if args.dataset_type == 'classification':
            args.metric = 'auc'
        elif args.dataset_type == 'multiclass':
            args.metric = 'cross_entropy'
        else:
            args.metric = 'rmse'

    if not ((args.dataset_type == 'classification' and args.metric in ['auc', 'prc-auc', 'accuracy']) or
            (args.dataset_type == 'regression' and args.metric in ['rmse', 'mae', 'mse', 'r2']) or
            (args.dataset_type == 'multiclass' and args.metric in ['cross_entropy', 'accuracy'])):
        raise ValueError(f'Metric "{args.metric}" invalid for dataset type "{args.dataset_type}".')

    args.minimize_score = args.metric in ['rmse', 'mae', 'mse', 'cross_entropy']

    update_checkpoint_args(args)

    if args.features_only:
        assert args.features_generator or args.features_path

    args.use_input_features = args.features_generator or args.features_path

    if args.features_generator is not None and 'rdkit_2d_normalized' in args.features_generator:
        assert not args.features_scaling

    args.num_lrs = 1

    if args.ffn_hidden_size is None:
        args.ffn_hidden_size = args.hidden_size

    assert (args.split_type == 'predetermined') == (args.folds_file is not None) == (args.test_fold_index is not None)
    assert (args.split_type == 'crossval') == (args.crossval_index_dir is not None)
    assert (args.split_type in ['crossval', 'index_predetermined']) == (args.crossval_index_file is not None)
    if args.split_type in ['crossval', 'index_predetermined']:
        with open(args.crossval_index_file, 'rb') as rf:
            args.crossval_index_sets = pickle.load(rf)
        args.num_runs = len(args.crossval_index_sets)
        args.seed = 0

    if args.test:
        args.epochs = 0

    # ============================================================
    # TAPT 参数验证和默认值设置
    # ============================================================
    if args.step == 'tapt' and not args.use_tapt:
        args.use_tapt = True

    if args.use_tapt:
        if args.step in ['functional_prompt', 'finetune_add', 'finetune_concat']:
            args.step = 'tapt'
        # ✅ 兼容性逻辑：如果用户用了旧的 --only_task 参数，自动转为 mode2
        if args.only_task:
            print("⚠️ Warning: --only_task is deprecated. Setting mode='task_only'.")
            args.mode = 'task_only'

        # 🆕 only_task 模式的参数冲突检查
        if args.only_task and args.enable_structure_prompt:
            import warnings
            warnings.warn(
                "⚠️ Conflicting flags detected:\n"
                "   --only_task forces noise-based mode\n"
                "   --enable_structure_prompt will be ignored\n"
                "   To use real structure features, remove --only_task flag."
            )
            # 强制覆盖
            args.enable_structure_prompt = False

        if not args.freeze_kano and args.kano_lr >= args.prompt_lr:
            import warnings
            warnings.warn(f'Warning: kano_lr ({args.kano_lr}) should typically be smaller than '
                          f'prompt_lr ({args.prompt_lr}) for effective differential learning.')

        # ... 其他验证逻辑保持不变 ...

        # ============================================================
        # 🆕 结构噪声控制开关信息输出
        # ============================================================
        if not args.quiet:
            print(f'\n{"=" * 60}')
            print(f'🚀 TAPT Mode Enabled')
            print(f'{"=" * 60}')

            # 🆕 显示运行模式
            if args.only_task:
                print(f'🔧 Mode: Task-Only Reproduction (KANO disabled)')
                print(f'   - Skipping KANO Functional Prompt')
                print(f'   - Using stochastic noise regularization')
                print(f'   - Compatible with previous BBBP/ToxCast results')
            else:
                print(f'✨ Mode: Full Model (KANO + TAPT)')
                print(f'   - Layer 1: KANO Functional Prompt (explicit structure)')
                print(f'   - Layer 2: TAPT Task Prompt (task adaptation)')

            print(f'Prompt Dimension: {args.prompt_dim}')
            print(f'Prompt Tokens per Task: {args.num_prompt_tokens}')
            print(f'KANO Learning Rate: {args.kano_lr} ({"frozen" if args.freeze_kano else "fine-tuning"})')
            print(f'Prompt Learning Rate: {args.prompt_lr}')
            print(f'Prompt Injection Layers: {args.prompt_injection_layers}')
            print(f'TAPT Dropout: {args.tapt_dropout}')
            print(f'Weight Decay: {args.weight_decay}')

            # ✅ 显示结构特征模式（仅在非 only_task 模式下有意义）
            if not args.only_task:
                if args.enable_structure_prompt:
                    print(f'🔧 TAPT Structure Mode: Real Features (Experimental)')
                else:
                    print(f'✨ TAPT Structure Mode: Stochastic Noise Regularization')
                print(f'   Structure Noise Scale: {args.structure_noise_scale}')

            print(f'{"=" * 60}\n')


def parse_train_args() -> Namespace:
    """
    Parses arguments for training (includes modifying/validating arguments).

    :return: A Namespace containing the parsed, modified, and validated args.
    """
    parser = ArgumentParser()
    add_train_args(parser)
    args = parser.parse_args()

    # 调用修改和验证函数
    modify_train_args(args)

    return args
