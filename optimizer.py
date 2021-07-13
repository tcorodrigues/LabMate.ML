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
from sklearn.model_selection import GridSearchCV
from joblib import dump

import initializer


parser = argparse.ArgumentParser()
parser.add_argument('-o', '--out_dir', type=str, action='store', default='output_files', help='dir to save files to.')
args = parser.parse_args()


output_dir = args.out_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print('Welcome! Let me work out what is the best experiment for you to run...')

'''
The training data should be a tab separated file named 'train_data.txt'. The first column of the file is the reaction identifier and the last column is the objective variable (target). The columns in between correspond to descriptors. Otherwise please change accordingly.
See example files
'''

filename = 'train_data.txt'
train = pd.read_csv(os.path.join(initializer.init_files_dir, filename), sep='\t')
array = train.values
X = array[:, 1:-1]
Y = array[:, -1]

'''
General settings below. These do not need to be changed.
The seed value is what makes the whole process deterministic. You may choose to change this number.
The possible number of estimators, max_features and max_depth is a good compromise, but may need to be adapted, if the number of features (columns) is very different.
'''

seed = 1
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
scoring = 'neg_mean_absolute_error'
model = RandomForestRegressor(random_state=seed)
estimators_int = list(range(100, 1050, 50))
param_grid = {'n_estimators': estimators_int, 'max_features': ('auto', 'sqrt'), 'max_depth': [None, 2, 4]}

print('All good till now. I am figuring out the best method to analyze your data. Bear with me...')

'''
This section makes LabMate.AI search for the best hyperparameters autonomously.
It will also save a file with the best score and store the ideal hyperparameters for future use.
'''

grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold, n_jobs=6)
grid_result = grid.fit(X, Y)
np.savetxt(os.path.join(output_dir, 'best_score.txt'), ["best_score: %s" % grid.best_score_], fmt='%s')
best_params = pd.DataFrame([grid.best_params_], columns=grid.best_params_.keys())

print('... done! It is going to be lightspeed from here on out! :)')

'''
This section loads all possible reactions (search space) and deletes all previously executed reactions from that file.
The file has the same format as the training data, but no "Target" column. Please check example file.
'''

filename2 = 'all_combos.txt'
df_all_combos = pd.read_csv(os.path.join(initializer.init_files_dir, filename2), sep='\t')
df_train_corrected = train.iloc[:, :-1]
unseen = pd.concat([df_all_combos, df_train_corrected]).drop_duplicates(keep=False)
array2 = unseen.values
X2 = array2[:, 1:]
df_all_combos2 = df_all_combos.iloc[:, 1:]

'''
LabMate.AI predicts the future in this section. It builds the model using the best hyperparameter set and predicts the reaction yield (numeric value) for each instance.
For your reference, the method creates a file with the feature importances
'''

model2 = RandomForestRegressor(n_estimators=grid.best_params_['n_estimators'],
                               max_features=grid.best_params_['max_features'], max_depth=grid.best_params_['max_depth'],
                               random_state=seed)
RF_fit = model2.fit(X, Y)
predictions = model2.predict(X2)
predictions_df = pd.DataFrame(data=predictions, columns=['Prediction'])
feat_imp = pd.DataFrame(model2.feature_importances_, index=list(df_all_combos2.columns.values),
                        columns=['Feature_importances'])
feat_imp = feat_imp.sort_values(by=['Feature_importances'], ascending=False)

'''
LabMate.AI calculates variances for the predictions, which allows prioritizing the next best experiment, and creates a table with all the generated information.
'''

variance = np.var([e.predict(X2) for e in model2.estimators_], axis=0)
variance_df = pd.DataFrame(data=variance, columns=['Variance'])

assert len(variance) == len(predictions)  # control line
initial_data = pd.DataFrame(data=array2, columns=list(unseen.columns.values))
df = pd.concat([initial_data, predictions_df, variance_df], axis=1)

'''
LabMate.AI now selects the next reaction to be performed.
'''

feat_imp_T = feat_imp.transpose()  # creates a table with a single row stating the importance (0-1 scale) of each variable
keys1 = list(feat_imp_T.keys())  # collects the names of the features
keys2 = list(feat_imp_T.keys())  # same as above
keys1.insert(7, 'Prediction')  # Inserts "Prediction" in position 7 of the previously generated list
keys2.insert(7, 'Variance')  # Inserts "Variance" in position 7 of the previously generated list

df_sorted = df.sort_values(by=[keys1[-1], keys1[0]], ascending=[False,
                                                                False])  # Fetches the table with the predictions and variance and sorts: 1) high prediction first; 2) most important feature second (descending order) for overlapping predictions
preliminary = df_sorted.iloc[0:5]  # Collects the first five columns
df_sorted2 = preliminary.sort_values(by=[keys2[-1], keys2[0]], ascending=[True,
                                                                          False])  # Sorts the top five rows by: 1) Low variance first; 2) most important feature second (descending order) for overlapping predictions
toPerform = df_sorted2.iloc[0]  # First row is the selected reaction

'''
Save files
'''

feat_imp.to_csv(os.path.join(output_dir, 'feature_importances.txt'), sep='\t')
best_params.to_csv(os.path.join(output_dir, 'best_parameters.txt'), sep='\t')
toPerform.to_csv(os.path.join(output_dir, 'selected_reaction.txt'), sep='\t')
df_sorted.to_csv(os.path.join(output_dir, 'predictions.txt'), sep='\t')
filename3 = os.path.join(output_dir, 'random_forest_model_grid.sav')
dump(grid, os.path.join(filename3))

print('You are all set! Have a good one, mate!')

'''
After performing the reaction simply edit the train_data.txt file with the reaction conditions used and target value, before running the script again. Enjoy and happy chemistry :)
'''
