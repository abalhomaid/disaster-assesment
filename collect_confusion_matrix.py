import argparse
import torch
import os
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# source activate tllib_metric


def main(args: argparse.Namespace):
    print(args)
    SWD = '_SWD'
    TRADE_OFF = '_trade_offs'
    METRICS_FILE = 'confusion_matrix.csv'

    MODELS = ['cdan', 'coral', 'dann', 'mmd']
    seeds = ['32']
    output = []
    
    for MODEL in MODELS:

        if MODEL == 'dann':
            lambda_s = ['0.01', '0']
            trade_offs = [ '1', '1']
        else:
            lambda_s = ['0']
            trade_offs = ['1']

        for seed in seeds:

            base_path = f'./train_jobs/cdan/logs/{MODEL}/seed_{seed}'

            experiments = []
            for s, trade_off in zip(lambda_s, trade_offs):
                experiments.extend([
                    'Damage_E2M' + SWD + '_' + s + TRADE_OFF + '_' + trade_off,
                ])

            print(experiments)
            dfs = []

            for exp in experiments:
                metrics_file = METRICS_FILE
                exp_path = os.path.join(base_path, exp)
                df = pd.read_csv(os.path.join(exp_path, metrics_file))
                dfs.append(df)

            dfs = pd.concat(dfs)
            dfs['model'] = MODEL
            dfs['seed'] = seed
            dfs = dfs[['model', 'seed', '0', '1']]

            df = dfs.copy(deep=True)

            # Format
            output.append(df)

    # Save
    output = pd.concat(output)
    print(output)
    output.to_csv(f'confusion_matrix_all.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='metrics collection')
    parser.add_argument('--swd', action='store_true', help='read from swd experiment')
    args = parser.parse_args()
    main(args)
