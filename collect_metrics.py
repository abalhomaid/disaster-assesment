import argparse
import torch
import os
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Source activate tllib_metric


def main(args: argparse.Namespace):
    print(args)
    if args.swd:
        print('Reading from SWD experiment')
        SWD = '_SWD'
    else:
        SWD = ''

    base_path = './train_jobs/cdan/logs/cdan'
    experiments = [
        'Damage_E2M' + SWD, 
        'Damage_E2N' + SWD, 
        'Damage_E2R' + SWD, 
        'Damage_M2E' + SWD,
        'Damage_M2N' + SWD,
        'Damage_M2R' + SWD, 
        'Damage_N2E' + SWD, 
        'Damage_N2M' + SWD, 
        'Damage_N2R' + SWD, 
        'Damage_R2E' + SWD, 
        'Damage_R2M' + SWD, 
        'Damage_R2N' + SWD
    ]

    dfs = []

    for exp in experiments:
        metrics_file = 'metrics_output.csv'
        exp_path = os.path.join(base_path, exp)
        df = pd.read_csv(os.path.join(exp_path, metrics_file))
        dfs.append(df)

    dfs = pd.concat(dfs)

    # Add source and target columns
    dfs['source'] = dfs['exp'].str[7]
    dfs['target'] = dfs['exp'].str[9]
    dfs = dfs[['source', 'target', 'accuracy', 'precision', 'recall', 'f1']]

    df = dfs.copy(deep=True)

    # Format
    df['accuracy'] = df['accuracy'].map('{:.3f}'.format)
    df['precision'] = df['precision'].map('{:.3f}'.format)
    df['recall'] = df['recall'].map('{:.3f}'.format)
    df['f1'] = df['f1'].map('{:.3f}'.format)

    # Save
    print(df)
    df.to_csv(f'metrics{SWD}.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='metrics collection')
    parser.add_argument('--swd', action='store_true', help='read from swd experiment')
    args = parser.parse_args()
    main(args)
