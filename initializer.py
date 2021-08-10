from itertools import product
import argparse
import os

import yaml
import pandas as pd
import numpy as np
import random
from multiprocessing import Pool

def several_combo_generator(input_randomstate, combo_counts):
    np.random.seed(input_randomstate)
    several_combo_list = []
    for i in range(0, combo_counts):
        several_combo_list.append(one_combo_generator())
    return several_combo_list

def one_combo_generator():
    one_combo_list = []
    for feature,boundaries_list in Dict_boundaries.items():
        lower = boundaries_list[0]
        upper = boundaries_list[1]
        random_per_feature = np.round(np.random.uniform(lower, upper), 2)
        one_combo_list.append(random_per_feature)
    return one_combo_list


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--init_dir', type=str, action='store', default=r'./init_files',
                    help='dir to save files to.')
parser.add_argument('-b', '--boundary', type=str, action='store', default='Boundaries.yml',
                    help='File containing boundary ranges.')
parser.add_argument('-s', '--seed', type=int, action='store', default=1,
                    help='Random seed for sampling.')
parser.add_argument('-n', '--n_samples', type=int, action='store', default=10,
                    help='Number conditions to sample.')
#choose to get all combo with grid search or random search
parser.add_argument('-m', '--method', type=str, action='store', default='grid',
                    help='Method for getting the all combo features.')
#random search feature sample number
parser.add_argument('-c', '--combo_num', type=int, action='store', default=20000,
                    help='Number combo to be generate in random search method.')
#multiprocessing pool number
parser.add_argument('-p', '--pool_num', type=int, action='store', default=5,
                    help='Number CPU to execute.')

args = parser.parse_args()

#Error messages for incorrect input argument
if args.n_samples < 10:
    raise ValueError('At least 10 initial samples are required in order for LabMate.ML to run.')

if args.method != 'grid' and  args.method != 'random':
    raise ValueError('Make sure you\'re entering either "grid" or "random" for the search method.')

if args.combo_num != 20000 and args.method != 'random':
    raise ValueError('Only when selecting "random" in "-m" argument, the combo number can be costumized.')

init_files_dir = args.init_dir
if not os.path.exists(init_files_dir):
    os.makedirs(init_files_dir)

with open(args.boundary, 'r') as f:
    boundaries = yaml.load(f, yaml.SafeLoader)

#Generate all_combo.txt and train_data.txt with random search algorithms.
if(args.method == 'random'):
    #Generate dictionary storing boundaries of each features.
    Dict_boundaries = {}
    for feature in boundaries.keys():
        lower_boundary = float(boundaries[feature][0])
        upper_boundary = float(boundaries[feature][-1])
    
        #If user didn't prepare their boundaries file in ascending format.
        if upper_boundary < lower_boundary:
            temp_boundary = lower_boundary
            lower_boundary = upper_boundary
            upper_boundary = temp_boundary
        Dict_boundaries[feature] = [lower_boundary, upper_boundary]
    
    #Set args argument to variable
    pool_num = args.pool_num
    combo_num = args.combo_num
    
    #Calculating how many combos per cpu have to be generated.
    combo_per_cpu = int(combo_num / pool_num)
    left_combo_cpu = combo_num % pool_num
    pool = Pool(pool_num)
    
    #Generate special pool input list for multiprocessing
    pool_input_list = []
    for j in range(0, pool_num):
        tuple_input_arg = (1 + j, combo_per_cpu)
        pool_input_list.append(tuple_input_arg)
    if (left_combo_cpu > 0):
        pool_input_list.append((pool_num + 1,left_combo_cpu))
    pool_outputs = pool.starmap(several_combo_generator, pool_input_list)
    
    #Merge all output list by every single cpu together into a final combo list.
    cpu_concat_list = []
    for one_cpu_combo_list in pool_outputs:
        df_temp = pd.DataFrame(one_cpu_combo_list)
        cpu_concat_list.append(df_temp)
    df_random_search = pd.concat(cpu_concat_list,axis=0, ignore_index=True)
    df_random_search.columns = boundaries.keys()
    df = df_random_search

#Generate all_combo.txt and train_data.txt with grid search algorithms.
else:
    df = pd.DataFrame(data=product(*boundaries.values()), columns=boundaries.keys())

df_random_data = df.sample(n=args.n_samples, random_state=args.seed)

df.to_csv(os.path.join(init_files_dir, 'all_combos.txt'), sep=',', index=False)
df_random_data.to_csv(os.path.join(init_files_dir, 'train_data.txt'), sep=',', index=False)