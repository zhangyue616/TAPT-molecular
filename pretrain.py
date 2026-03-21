"""
Pre-training script for TAPT graph encoder.
Usage: python scripts/pretrain.py --data_path <path> --save_dir <path> [options]
"""
import warnings

warnings.filterwarnings('ignore')
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

import argparse
from argparse import Namespace
from chemprop.train.run_training import pre_training
from chemprop.parsing import parse_train_args, modify_train_args
from chemprop.torchlight import initialize_exp


def pretrain(args: Namespace):
    logger, args.save_dir = initialize_exp(Namespace(**args.__dict__))
    pre_training(args, logger)
    print(f"Pre-training complete. Model saved to {args.save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-train TAPT graph encoder')
    parser.add_argument('--data_path', default='./data/zinc15_250K.csv',
                        help='Path to pre-training dataset')
    parser.add_argument('--save_dir', default='./dumped/pretrained_encoder',
                        help='Directory to save checkpoints')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')

    custom_args = parser.parse_args()

    args = parse_train_args([])
    args.data_path = custom_args.data_path
    args.save_dir = custom_args.save_dir
    args.gpu = custom_args.gpu
    args.epochs = custom_args.epochs
    args.batch_size = custom_args.batch_size

    if custom_args.learning_rate:
        args.init_lr = custom_args.learning_rate

    modify_train_args(args)
    pretrain(args)
