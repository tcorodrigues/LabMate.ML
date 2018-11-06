import numpy as np
import pandas as pd
import itertools


'''
Below create lists for your reaction parameters. Change names of lists where appropriate
'''

#For bigger lists use np.arange(min_value, max_value, step)
Pyridine = [0.1, 0.2, 0.3] # in mmol
Aldehyde = [0.1, 0.2, 0.3] # in mmol
Isocyanide = [0.1, 0.2, 0.3] # in mmol
Temperature = [10, 20, 40, 60, 80] # in C
Solvent = [0.1, 0.25, 0.5, 1, 1.5] # in mL
Catalyst = [0, 1, 2, 3, 4, 5, 7.5, 10] # in mol%
Time = [5, 10, 15, 30, 60] # in minutes



'''
The following lines create all combos possible for the values listed above and saves as text file. Change names where appropriate.
'''
combinations = list(itertools.product(Pyridine, Aldehyde, Isocyanide, Temperature, Solvent, Catalyst, Time))
df = pd.DataFrame(combinations)
df.to_csv('all_combos.txt', sep = '\t', header = ['Pyridine', 'Aldehyde', 'Isocyanide', 'Temperature', 'Solvent', 'Catalyst', 'Time'])


'''
Below, 10 random reaction are selected from the all combinations table. The reactions are stored in a text file. Change names of header as appropriate.
'''

random_data = df.sample(n=10, random_state=1)
df_random_data = pd.DataFrame(random_data)
df_random_data.to_csv('train_data.txt', sep= '\t', header = ['Pyridine', 'Aldehyde', 'Isocyanide', 'Temperature', 'Solvent', 'Catalyst', 'Time'])