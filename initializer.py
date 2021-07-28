from itertools import product
import argparse
import os

import yaml
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--init_dir', type=str, action='store', default=r'./init_files',
                    help='dir to save files to.')
parser.add_argument('-b', '--boundary', type=str, action='store', default='Boundaries.yml',
                    help='File containing boundary ranges.')
parser.add_argument('-s', '--seed', type=int, action='store', default=1,
                    help='Random seed for sampling.')
parser.add_argument('-n', '--n_samples', type=int, action='store', default=10,
                    help='Number conditions to sample.')
args = parser.parse_args()

if args.n_samples < 10:
    raise ValueError('At least 10 initial samples are required in order for LabMate.ML to run.')

init_files_dir = args.init_dir
if not os.path.exists(init_files_dir):
    os.makedirs(init_files_dir)

with open(args.boundary, 'r') as f:
    boundaries = yaml.load(f, yaml.SafeLoader)

df = pd.DataFrame(data=product(*boundaries.values()), columns=boundaries.keys())
df_random_data = df.sample(n=args.n_samples, random_state=args.seed)

df.to_csv(os.path.join(init_files_dir, 'all_combos.txt'), sep=',')
df_random_data.to_csv(os.path.join(init_files_dir, 'train_data.txt'), sep=',')
