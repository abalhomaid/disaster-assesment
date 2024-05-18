import argparse
import torch
import os
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np


# source activate tllib_metric


def main(args: argparse.Namespace):
    print(args)
    SWD = '_SWD'
    TRADE_OFF = '_trade_offs'
    METRICS_FILE = 'metrics_output.csv'
    # MODEL = 'coral'
    # MODEL = 'mmd'
    # MODEL = 'dann'
    # df = pd.read_csv(f'metrics{SWD}_lambda_s{TRADE_OFF}_{MODEL}.csv')
    # df = pd.read_csv(f'metrics_all.csv')

    # # # df = df.groupby(['source', 'target']).agg({'f1': 'max'})
    # # # idxmax = df.groupby(['seed', 'source', 'target'])['f1'].idxmax()
    # # # df = df.loc[idxmax]

    # # # Other models
    # df = df[(df['swd'] == 0)]
    # df = df.groupby(['model', 'source', 'target'])[['accuracy', 'precision', 'recall', 'f1']].mean().reset_index()

    # df = df.pivot(index=['source', 'target'], columns=['model']).reset_index().round(3)
    # df.columns = df.columns.swaplevel(0, 1)
    # df.sort_index(axis=1, level=0, inplace=True)
    # df = df.reindex(columns=['accuracy', 'precision', 'recall', 'f1'], level=1)
    # print(df.columns)
    # # Save
    # print(df)
    # df.to_csv(f'metrics_all_filtered_swapped.csv', index=False)
    # df.to_csv(f'metrics_all_filtered.csv', index=False)

    # AWDANN
    # df = df[df['model'] == 'dann']
    # df = df[df['swd'] != 0]
    # idxmax = df.groupby(['seed', 'source', 'target'])['f1'].idxmax()
    # df = df.loc[idxmax]
    # swds = df['swd'].values
    # print(df)
    # df = df.groupby(['model', 'source', 'target'])[['accuracy', 'precision', 'recall', 'f1']].mean().round(3).reset_index()
    # print(df)
    # df.to_csv(f'metrics_awdann_filtered.csv', index=False)

    # Stats
    # print(df.groupby(['model', 'seed']).count())

    # Format for paper
    df = pd.read_csv('metrics_all_filtered_swapped.csv', header=[0, 1])
    to_latex_rows(df)
    print(df)

    df = pd.read_csv('metrics_awdann_filtered.csv', header=0)
    source_domain_count = 3
    
    for i in range(0, len(df), source_domain_count):
        print('AWDANN' + ' & ' + np.array2string(df[['accuracy', 'precision', 'recall', 'f1']].loc[i:i+source_domain_count-1].values.flatten(), separator=" & ")[1:-1] + ' \\\\')
        print()

def to_latex_rows(df):
    models = ['cdan', 'coral', 'dann', 'mmd']
    source_domain_count = 3
    for i in range(0, len(df), source_domain_count):
        for model in models:
            print(model.upper() + ' & ' + np.array2string(df[model].loc[i:i+source_domain_count-1].values.flatten(), separator=" & ")[1:-1] + ' \\\\')
        print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='metrics collection')
    parser.add_argument('--swd', action='store_true', help='read from swd experiment')
    args = parser.parse_args()
    main(args)
