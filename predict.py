"""
Prediction script for trained TAPT models.
Usage: python scripts/predict.py --checkpoint <path> --data_path <path> --output <path>
"""
import argparse
import warnings
import pandas as pd
from pathlib import Path
from chemprop.parsing import parse_train_args, modify_train_args
from chemprop.train import make_predictions

warnings.filterwarnings('ignore')


def predict(checkpoint_path, data_path, output_path, dataset_type='classification', num_tasks=1):
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    args = parse_train_args([])
    args.checkpoint_path = checkpoint_path
    args.num_tasks = num_tasks
    args.dataset_type = dataset_type
    args.test_path = data_path
    modify_train_args(args)

    data = pd.read_csv(data_path)
    if 'smiles' not in data.columns:
        raise ValueError(f"Data must contain 'smiles' column. Found: {data.columns.tolist()}")

    pred, smiles = make_predictions(args, data.smiles.tolist())

    df_result = pd.DataFrame({'smiles': smiles})
    if isinstance(pred[0], list):
        for i in range(len(pred[0])):
            df_result[f'pred_{i}'] = [item[i] for item in pred]
    else:
        df_result['prediction'] = pred

    df_result.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    return df_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make predictions with trained TAPT model')
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint path')
    parser.add_argument('--data_path', required=True, help='Input CSV with smiles column')
    parser.add_argument('--output', default='./predictions.csv', help='Output CSV path')
    parser.add_argument('--dataset_type', default='classification',
                        choices=['classification', 'regression'])
    parser.add_argument('--num_tasks', type=int, default=1, help='Number of tasks')

    args = parser.parse_args()
    predict(args.checkpoint, args.data_path, args.output, args.dataset_type, args.num_tasks)
