import argparse
import torch
import os
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import glob
import numpy as np

# source activate tllib_metric


def main(args: argparse.Namespace):
    print(args)
    SWD = '_SWD'
    TRADE_OFF = '_trade_offs'
    METRICS_FILE = 'analysis-*.txt'

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

            dfs = []

            for exp in experiments:
                metrics_file = METRICS_FILE
                exp_path = os.path.join(base_path, exp)
                # df = pd.read_csv(os.path.join(exp_path, metrics_file))
                # dfs.append(df)
                
                file_list = glob.glob(os.path.join(exp_path, metrics_file))

                latest_timestamp = 0
                # Iterate over each file
                for file_path in file_list:
                    # Get the modification timestamp of the current file
                    timestamp = os.path.getmtime(file_path)
                    # Check if the current file has a later timestamp than the latest one found so far
                    if timestamp > latest_timestamp:
                        latest_timestamp = timestamp
                        latest_file = file_path

                with open(latest_file, 'r') as file:
                    last_line = file.readlines()[-1].strip()
                    # Process the contents as needed
                    print([MODEL, seed, exp, last_line])
                    
                    start_adistance = 7
                    end_adistance = 10    

                    if '0.01' in exp:
                        dfs.append(['awdan', seed, exp[start_adistance:end_adistance], last_line])
                    else:
                        dfs.append([MODEL, seed, exp[start_adistance:end_adistance], last_line])
                    

     
            
            dfs = pd.DataFrame(dfs)
            print(dfs)
            # dfs['model'] = MODEL
            # dfs['seed'] = seed
            # dfs = dfs[['model', 'seed', '0', '1']]

            # df = dfs.copy(deep=True)
            # df.columns = ['model', 'seed', 'exp', 'A-distance']
            # df = pd.DataFrame(dfs, columns=['model', 'seed', 'exp', 'A-distance'])
            # df['A-distance'] = df['A-distance'].str.slice(20, 27)
            # Format
            output.append(dfs)

    # Save
    output = pd.concat(output)
    output.columns = ['model', 'seed', 'exp', 'A-distance']
    output['A-distance'] = output['A-distance'].str.strip()
    
    # A-model
    # output['A-distance'] = output['A-distance'].str.slice(20, 26)

    # SVM
    output['A-distance'] = output['A-distance'].str.slice(13)
    output.to_csv(f'adistance_pre.csv', index=False)

    # A-distance is 0
    output['A-distance'] = output['A-distance'].replace('0., de', '0')

    output['A-distance'] = output['A-distance'].astype(float)
    output['A-distance'] = output['A-distance'].round(3)
    print(output)

    output = output.pivot(index='model', columns='exp', values='A-distance')
    print(output)
    to_latex(output)
    output.to_csv(f'adistance_all.csv', index=False)


def to_latex(output):
    for i, row in enumerate(output.values):
        print(output.index[i].upper() + ' & ' + np.array2string(row, separator=" & ")[1:-1] + ' & ' + str(row.mean().round(3)) + ' \\\\')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='metrics collection')
    parser.add_argument('--swd', action='store_true', help='read from swd experiment')
    args = parser.parse_args()
    main(args)
