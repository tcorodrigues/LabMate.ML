"""
Please read the license file.
LabMate.AI was designed to help identifying optimized conditions for chemical reactions.
You will need the Python libraries below (NumPy, Pandas and Scikit-learn) and 10 random reactions to run LabMate.AI.
"""

import os
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from joblib import dump


parser = argparse.ArgumentParser()
parser.add_argument('-o', '--out_dir', type=str, action='store', default='output_files', help='dir to save files to.')
parser.add_argument('-t', '--train_file', type=str, action='store', default='train_data.txt', help='Training data.')
parser.add_argument('-i', '--init_dir', type=str, action='store', default=r'./init_files', help='dir to load files from.')
parser.add_argument('-s', '--seed', type=int, action='store', default=1, help='Random seed value.')
parser.add_argument('-m', '--metric', type=str, action='store', default='neg_mean_absolute_error', help='Metric for evaluatng hyperparameters.')
parser.add_argument('-c', '--combos_file', type=str, action='store', default='all_combos.txt', help='File containing all reaction combinations.')
parser.add_argument('-j', '--jobs', type=int, action='store', default=6, help='Number of parallel jobs when optimising hyperparameters.')
parser.add_argument('-g', '--grid', type=str, action='store', default='grid', help='Grid methods for tuning the hyperparameters.')
args = parser.parse_args()


output_dir = args.out_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print('Welcome! Let me work out what is the best experiment for you to run...')

'''
The training data should be a tab separated file named.
The first column of the file is the reaction identifier and the last column is the objective variable (target).
The columns in between correspond to descriptors.
Otherwise please change accordingly.
See example files
'''

train = pd.read_csv(os.path.join(args.init_dir, args.train_file), sep=',')
array = train.values
X = array[:, :-1]
Y = array[:, -1]

'''
General settings below. These do not need to be changed.
The seed value is what makes the whole process deterministic.
You may choose to change this number.
The possible number of estimators, max_features and max_depth is a good compromise, but may need to be adapted,
if the number of features (columns) is very different.
'''

seed = args.seed
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
model = RandomForestRegressor(random_state=seed)
#estimators_int = list(range(100, 1050, 50))
#param_grid = {'n_estimators': estimators_int, 'max_features': ('auto', 'sqrt'), 'max_depth': [None, 2, 4]}

print('All good till now. I am figuring out the best method to analyze your data. Bear with me...')
print('Looking for the best method by ' + args.grid + ' search...' )

'''
This section makes LabMate.AI search for the best hyperparameters autonomously.
It will also save a file with the best score and store the ideal hyperparameters for future use.
'''

#Grid search and its parameters gird set preparation
if args.grid == 'grid':
    estimators_int = list(range(100, 1050, 50))
    param_grid = {'n_estimators': estimators_int, 'max_features': ('auto', 'sqrt'), 'max_depth': [None, 2, 4]}
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=args.metric, cv=kfold, n_jobs=args.jobs)

#Random search and its parameters grid set preparation
elif args.grid == 'random':
    param_grid = {'bootstrap': [True, False],
                  'max_features': ['auto', 'sqrt', 'log2'],
                  'max_depth': [2,3,4,5,6,7,8,9,10],
                  'n_estimators': list(range(10, 201, 5))}

    grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, scoring=args.metric, n_iter=20, cv=kfold, n_jobs=args.jobs)


#start grid/radom search tuning
grid_result = grid.fit(X, Y)
np.savetxt(os.path.join(output_dir, 'best_score.txt'), ["best_score: %s" % grid.best_score_], fmt='%s')
best_params = pd.DataFrame([grid.best_params_], columns=grid.best_params_.keys())


print('... done! It is going to be lightspeed from here on out! :)')



'''
This section loads all possible reactions (search space) and deletes all previously executed reactions from that file.
The file has the same format as the training data, but no "Target" column.
Please check example file.
'''

df_all_combos = pd.read_csv(os.path.join(args.init_dir, args.combos_file), sep=',')
unseen = pd.concat([df_all_combos, train.iloc[:, :-1]]).drop_duplicates(keep=False)
X2 = unseen.values

'''
LabMate.AI predicts the future in this section.
It builds the model using the best hyperparameter set and predicts the reaction yield (numeric value) for each instance.
For your reference, the method creates a file with the feature importances
'''

model2 = grid.best_estimator_  # extracts best estimator (equivalent to setting params), seed is conserved.
predictions = model2.predict(X2)
predictions_df = pd.DataFrame(data=predictions, columns=['Prediction'])
feat_imp = pd.DataFrame(model2.feature_importances_, index=list(df_all_combos.columns.values),
                        columns=['Feature_importances'])
feat_imp = feat_imp.sort_values(by=['Feature_importances'], ascending=False)

'''
LabMate.AI calculates variances for the predictions, which allows prioritizing the next best experiment, and
creates a table with all the generated information.
'''

variance = np.var([e.predict(X2) for e in model2.estimators_], axis=0)
variance_df = pd.DataFrame(data=variance, columns=['Variance'])

assert len(variance) == len(predictions)  # control line
initial_data = pd.DataFrame(data=X2, columns=list(unseen.columns.values))
df = pd.concat([initial_data, predictions_df, variance_df], axis=1)

'''
LabMate.AI now selects the next reaction to be performed.
'''

feat_imp_T = feat_imp.transpose()  # creates a table with a single row stating the importance (0-1 scale) of each variable
keys1 = list(feat_imp_T.keys())  # collects the names of the features
keys2 = list(feat_imp_T.keys())  # same as above
keys1.append('Prediction')  # Inserts "Prediction" in the end of previously generated list
keys2.append('Variance')  # Inserts "Variance" in the end of the previously generated list

df_sorted = df.sort_values(by=[keys1[-1], keys1[0]], ascending=[False,
                                                                False])  # Fetches the table with the predictions and variance and sorts: 1) high prediction first; 2) most important feature second (descending order) for overlapping predictions
preliminary = df_sorted.iloc[0:5]  # Collects the first five columns
df_sorted2 = preliminary.sort_values(by=[keys2[-1], keys2[0]], ascending=[True,
                                                                          False])  # Sorts the top five rows by: 1) Low variance first; 2) most important feature second (descending order) for overlapping predictions
toPerform = df_sorted2.iloc[0]  # First row is the selected reaction

'''
Save files
'''

feat_imp.to_csv(os.path.join(output_dir, 'feature_importances.txt'), sep=',')
best_params.to_csv(os.path.join(output_dir, 'best_parameters.txt'), sep=',', index=False)
toPerform.to_csv(os.path.join(output_dir, 'selected_reaction.txt'), sep=',')
with open(os.path.join(output_dir, 'selected_reaction.txt'), 'r') as fin:
    data = fin.read().splitlines(True)
with open(os.path.join(output_dir, 'selected_reaction.txt'), 'w') as fout:
    fout.writelines(data[1:])
toPerform[:-2].to_frame().transpose().to_csv(os.path.join(args.init_dir, 'train_data.txt'), mode='a', header=False, index=False)
df_sorted.to_csv(os.path.join(output_dir, 'predictions.txt'), sep=',', index=False)
filename3 = os.path.join(output_dir, 'random_forest_model_grid.sav')
dump(grid, os.path.join(filename3))

print('You are all set! Have a good one, mate!')

print('''
After performing the reaction simply edit the training data file with the reaction conditions used and target value,
before running the script again. Enjoy and happy chemistry :)
''')
