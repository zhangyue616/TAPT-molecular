from argparse import ArgumentParser, Namespace
import json
import os
import torch
from chemprop.utils import makedirs
from chemprop.features import get_available_features_generators


def add_predict_args(parser: ArgumentParser):
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
                        help='Directory from which to load model checkpoints (walks directory and ensembles all models that are found)',
                        default=None)
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
    add_predict_args(parser)

    parser.add_argument('--data_path', type=str,
                        help='Path to data CSV file',
                        default='data.csv')
    parser.add_argument('--dataset_type', type=str,
                        choices=['classification', 'regression', 'multiclass'],
                        help='Type of dataset',
                        default='regression')
    parser.add_argument('--multiclass_num_classes', type=int, default=3,
                        help='Number of classes when using multiclass dataset')
    parser.add_argument('--save_dir', type=str, default='./ckpt',
                        help='Directory where model checkpoints will be saved')
    parser.add_argument('--separate_val_path', type=str, default=None,
                        help='Path to separate val set')
    parser.add_argument('--separate_val_features_path', type=str, nargs='*', default=None,
                        help='Path to file with features for separate val set')
    parser.add_argument('--separate_test_path', type=str, default=None,
                        help='Path to separate test set')
    parser.add_argument('--separate_test_features_path', type=str, nargs='*', default=None,
                        help='Path to file with features for separate test set')
    parser.add_argument('--split_type', type=str, default='random',
                        choices=['random', 'scaffold_balanced', 'predetermined', 'crossval', 'index_predetermined',
                                 'cluster_balanced'],
                        help='Method of splitting the data into train/val/test')
    parser.add_argument('--split_sizes', type=float, nargs=3, default=[0.8, 0.1, 0.1],
                        help='Split proportions for train/validation/test sets')
    parser.add_argument('--num_runs', type=int, default=1,
                        help='Number of runs when training a model')
    parser.add_argument('--folds_file', type=str, default=None,
                        help='Optional file of folds to use for splitting')
    parser.add_argument('--val_fold_index', type=int, default=None,
                        help='Which fold to use as val for crossval splitting')
    parser.add_argument('--test_fold_index', type=int, default=None,
                        help='Which fold to use as test for crossval splitting')
    parser.add_argument('--crossval_index_dir', type=str,
                        help='Directory in which to find cross validation index files')
    parser.add_argument('--crossval_index_file', type=str,
                        help='Indices of files to use as train/val/test. Overrides --crossval_index_dir')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed to use when splitting data')
    parser.add_argument('--metric', type=str, default=None,
                        choices=['auc', 'prc-auc', 'rmse', 'mae', 'mse', 'r2', 'accuracy', 'cross_entropy'],
                        help='Metric to use during evaluation')
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='Skip non-essential print statements')
    parser.add_argument('--log_frequency', type=int, default=10,
                        help='The number of batches between each logging of the training loss')
    parser.add_argument('--show_individual_scores', action='store_true', default=False,
                        help='Show all scores for individual runs, not just the average')
    parser.add_argument('--no_cache', action='store_true', default=False,
                        help='Turn off caching mol2graph computation')
    parser.add_argument('--config_path', type=str,
                        help='Path to a .json file containing arguments')

    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to run')
    parser.add_argument('--warmup_epochs', type=float, default=2.0,
                        help='Number of epochs during which learning rate increases linearly')
    parser.add_argument('--init_lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--max_lr', type=float, default=1e-3,
                        help='Maximum learning rate')
    parser.add_argument('--final_lr', type=float, default=1e-4,
                        help='Final learning rate')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Temperature')

    parser.add_argument('--encoder_name', type=str, default='CMPNN', choices=['CMPNN', 'MPNN'],
                        help='Name of the encoder to use')
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
    parser.add_argument('--features_only', action='store_true', default=False,
                        help='Use only the additional features in an FFN, no graph network')
    parser.add_argument('--save_smiles_splits', action='store_true', default=False,
                        help='Save smiles for each train/val/test split')
    parser.add_argument('--test', action='store_true', default=False,
                        help='Whether to skip training and only test the model')
    parser.add_argument('--dump_path', default='dumped', type=str,
                        help='Dump path')
    parser.add_argument('--exp_name', default='test', type=str,
                        help='Experiment name')
    parser.add_argument('--exp_id', default='1', type=str,
                        help='Experiment ID')

    parser.add_argument('--step', default='functional_prompt', type=str,
                        help='Training step')
    parser.add_argument('--use_tapt', action='store_true', default=False,
                        help='Enable TAPT (Task-Aware Prompting). Defaults to Task-Only mode if --combine is not set.')
    parser.add_argument('--combine', action='store_true', default=False,
                        help='Enable Full Synergy (KANO + KG + Task). Overrides --use_tapt.')
    parser.add_argument('--only_task', action='store_true', default=False,
                        help='Deprecated')
    parser.add_argument('--tapt_alpha', type=float, default=0.001,
                        help='Weight of the TAPT prompt')
    parser.add_argument('--structure_noise_scale', type=float, default=0.01,
                        help='Structure noise scale')
    parser.add_argument('--task_pool_size', type=int, default=3,
                        help='Task pool size')
    parser.add_argument('--prompt_dim', type=int, default=128,
                        help='Prompt dimension')
    parser.add_argument('--num_prompt_tokens', type=int, default=5,
                        help='Number of prompt tokens')
    parser.add_argument('--kano_lr', type=float, default=1e-5,
                        help='KANO learning rate')
    parser.add_argument('--prompt_lr', type=float, default=1e-3,
                        help='Prompt learning rate')
    parser.add_argument('--task_id', type=int, default=0,
                        help='Task ID')
    parser.add_argument('--freeze_kano', action='store_true', default=False,
                        help='Freeze the KANO backbone')
    parser.add_argument('--prompt_injection_layers', type=int, nargs='+', default=None,
                        help='Layers to inject prompt')
    parser.add_argument('--tapt_dropout', type=float, default=0.1,
                        help='TAPT dropout')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay')
    parser.add_argument('--fg_dim', type=int, default=133,
                        help='Functional group dimension')


def update_checkpoint_args(args: Namespace):
    if hasattr(args, 'checkpoint_paths') and args.checkpoint_paths is not None:
        return

    if args.checkpoint_dir is not None and args.checkpoint_path is not None:
        raise ValueError('Only one of checkpoint_dir and checkpoint_path can be specified.')

    if args.checkpoint_path is not None:
        args.checkpoint_paths = [args.checkpoint_path]
        return

    if args.checkpoint_dir is not None:
        args.checkpoint_paths = []
        for root, _, files in os.walk(args.checkpoint_dir):
            for fname in files:
                if fname.endswith('.pt'):
                    args.checkpoint_paths.append(os.path.join(root, fname))
        if len(args.checkpoint_paths) == 0:
            raise ValueError(f'Failed to find any model checkpoints in directory "{args.checkpoint_dir}"')
        args.ensemble_size = len(args.checkpoint_paths)
        return

    args.checkpoint_paths = None


def modify_predict_args(args: Namespace):
    assert args.test_path
    assert args.preds_path
    assert args.checkpoint_dir is not None or args.checkpoint_path is not None or args.checkpoint_paths is not None
    update_checkpoint_args(args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    del args.no_cuda
    makedirs(args.preds_path, isfile=True)


def parse_predict_args() -> Namespace:
    parser = ArgumentParser()
    add_predict_args(parser)
    args = parser.parse_args()
    modify_predict_args(args)
    return args


def modify_train_args(args: Namespace):
    if args.config_path is not None:
        with open(args.config_path) as f:
            config = json.load(f)
            for key, value in config.items():
                setattr(args, key, value)

    if hasattr(args, 'exp_name') and hasattr(args, 'exp_id') and args.exp_name and args.exp_id:
        if args.save_dir == './ckpt':
            args.save_dir = os.path.join(args.dump_path, args.exp_name, args.exp_id)

    makedirs(args.save_dir)

    assert args.data_path is not None
    assert args.dataset_type is not None

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    del args.no_cuda

    args.features_scaling = not args.no_features_scaling
    del args.no_features_scaling

    if args.metric is None:
        if args.dataset_type == 'classification':
            args.metric = 'auc'
        elif args.dataset_type == 'multiclass':
            args.metric = 'cross_entropy'
        else:
            args.metric = 'rmse'

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

    if args.combine:
        args.use_tapt = True
        args.step = 'functional_prompt'
        print("\n🚀 [Config] Activated: Full Synergy (KANO + KG + Task)\n")
    elif args.use_tapt:
        args.step = 'pretrain'
        print("\n🚀 [Config] Activated: Task-Only (Structure Noise)\n")
    else:
        args.use_tapt = False
        args.step = 'functional_prompt'
        print("\n🚀 [Config] Activated: Baseline (Pure KANO)\n")

    if not args.freeze_kano and args.kano_lr >= args.prompt_lr and args.use_tapt:
        import warnings
        warnings.warn(f'Warning: kano_lr ({args.kano_lr}) >= prompt_lr ({args.prompt_lr})')


def parse_train_args() -> Namespace:
    parser = ArgumentParser()
    add_train_args(parser)
    args = parser.parse_args()
    modify_train_args(args)
    return args