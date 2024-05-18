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
    METRICS_FILE = 'metrics_output.csv'
    # MODEL = 'coral'
    # MODEL = 'mmd'
    # MODEL = 'dann'

    # lambda_s = ['0.2', '0.35', '0.5']
    # lambda_s = ['1000', '100', '10', '1', '0.1', '0.01', '0.001']
    
    # lambda_s = ['1', '1', '1', '1', '0.1', '0.01', '0.001']
    # trade_offs = ['0.001', '0.01', '0.1', '1', '1', '1', '1']

    MODELS = ['cdan', 'coral', 'dann', 'mmd']
    seeds = ['2', '32', '128']
    output = []
    
    for MODEL in MODELS:

        if MODEL == 'dann':
            lambda_s = ['1', '1', '1', '1', '0.1', '0.01', '0.001', '0']
            trade_offs = ['0.001', '0.01', '0.1', '1', '1', '1', '1', '1']
        else:
            lambda_s = ['0']
            trade_offs = ['1']

        for seed in seeds:

            base_path = f'./train_jobs/cdan/logs/{MODEL}/seed_{seed}'

            experiments = []
            for s, trade_off in zip(lambda_s, trade_offs):
                experiments.extend([
                    'Damage_E2M' + SWD + '_' + s + TRADE_OFF + '_' + trade_off,
                    'Damage_E2N' + SWD + '_' + s + TRADE_OFF + '_' + trade_off,
                    'Damage_E2R' + SWD + '_' + s + TRADE_OFF + '_' + trade_off,
                    'Damage_M2E' + SWD + '_' + s + TRADE_OFF + '_' + trade_off,
                    'Damage_M2N' + SWD + '_' + s + TRADE_OFF + '_' + trade_off,
                    'Damage_M2R' + SWD + '_' + s + TRADE_OFF + '_' + trade_off,
                    'Damage_N2E' + SWD + '_' + s + TRADE_OFF + '_' + trade_off,
                    'Damage_N2M' + SWD + '_' + s + TRADE_OFF + '_' + trade_off,
                    'Damage_N2R' + SWD + '_' + s + TRADE_OFF + '_' + trade_off,
                    'Damage_R2E' + SWD + '_' + s + TRADE_OFF + '_' + trade_off,
                    'Damage_R2M' + SWD + '_' + s + TRADE_OFF + '_' + trade_off,
                    'Damage_R2N' + SWD + '_' + s + TRADE_OFF + '_' + trade_off
                ])

            print(experiments)
            dfs = []

            for exp in experiments:
                metrics_file = METRICS_FILE
                exp_path = os.path.join(base_path, exp)
                df = pd.read_csv(os.path.join(exp_path, metrics_file))
                dfs.append(df)

            dfs = pd.concat(dfs)

            # Add source and target columns
            dfs['source'] = dfs['exp'].str[7]
            dfs['target'] = dfs['exp'].str[9]
            lambda_s_position = 3
            trade_offs_position = 6
            dfs['lambda_s'] = dfs['exp'].str.split('_', expand=True)[lambda_s_position]
            dfs['trade_off'] = dfs['exp'].str.split('_', expand=True)[trade_offs_position]
            dfs['swd'] = dfs['lambda_s'].astype(float) / dfs['trade_off'].astype(float)
            dfs['model'] = MODEL
            dfs['seed'] = seed
            dfs = dfs[['source', 'target', 'model', 'seed', 'swd', 'accuracy', 'precision', 'recall', 'f1']]

            df = dfs.copy(deep=True)

            # Format
            df['accuracy'] = df['accuracy'].map('{:.3f}'.format)
            df['precision'] = df['precision'].map('{:.3f}'.format)
            df['recall'] = df['recall'].map('{:.3f}'.format)
            df['f1'] = df['f1'].map('{:.3f}'.format)
            output.append(df)

    # Save
    output = pd.concat(output)
    print(output)
    # df.to_csv(f'metrics{SWD}_lambda_s{TRADE_OFF}_{MODEL}.csv', index=False)
    output.to_csv(f'metrics_all.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='metrics collection')
    parser.add_argument('--swd', action='store_true', help='read from swd experiment')
    args = parser.parse_args()
    main(args)
