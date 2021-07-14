import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from joblib import dump

import os

init_files_dir = r'../init_files'

if not os.path.exists('output_files'):
    os.makedirs('output_files')

#load data
filename = 'train_data.txt'
train = pd.read_csv(os.path.join(init_files_dir, filename), sep= ',')
array = train.values
X = array[:,:-1]
Y = array[:,-1]

#General stuff
seed = 1234
kfold = KFold(n_splits = 10, shuffle=True, random_state=seed)
scoring = 'neg_mean_absolute_error'
model = RandomForestRegressor(random_state=seed)

#Parameters to tune
estimators = np.arange(100, 1050, 50)
estimators_int = np.ndarray.tolist(estimators)
param_grid = {'n_estimators':estimators_int, 'max_features':('auto', 'sqrt'), 'max_depth':[None, 2, 4]}

#search best parameters and train
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold, n_jobs=6)
grid_result = grid.fit(X, Y)

#print the best data cranked out from the grid search
np.savetxt(os.path.join('output_files', 'best_score.txt'), ["best_score: %s" % grid.best_score_], fmt ='%s')
best_params = pd.DataFrame([grid.best_params_], columns=grid.best_params_.keys())

#Predict the future
filename2 = 'all_combos.txt'
df_all_combos = pd.read_csv(os.path.join(init_files_dir, filename2), sep= ',')
df_train_corrected = train.iloc[:,:-1]
unseen = pd.concat([df_all_combos, df_train_corrected]).drop_duplicates(keep=False)
X2 = unseen.values


model2 = RandomForestRegressor(n_estimators = grid.best_params_['n_estimators'], max_features = grid.best_params_['max_features'], max_depth = grid.best_params_['max_depth'], random_state = seed)
RF_fit = model2.fit(X, Y)
predictions = model2.predict(X2)
predictions_df = pd.DataFrame(data=predictions, columns=['Prediction'])
feat_imp = pd.DataFrame(model2.feature_importances_, index=['Pyridine', 'Aldehyde', 'Isocyanide', 'Temperature', 'Solvent', 'Catalyst','Time'], columns=['Feature_importances'])

#get individual tree preds
all_predictions = []
for e in model2.estimators_:
    all_predictions += [e.predict(X2)]

#get variance and dataframe
variance = np.var(all_predictions, axis=0)
variance_df = pd.DataFrame(data=variance, columns=['Variance'])

assert len(variance) == len(predictions)

#concatenate tables
initial_data = pd.DataFrame(data=X2, columns = ['Pyridine', 'Aldehyde', 'Isocyanide', 'Temperature', 'Solvent', 'Catalyst', 'Time'])
df = pd.concat([initial_data, predictions_df, variance_df], axis=1)

if len(Y) < 19:
	df_sorted = df.sort_values(by=['Variance', 'Catalyst'], ascending=[False, True])
	toPerform = df_sorted.iloc[0]

elif len(Y) >= 19 and np.max(Y[10:]) >= 4 * np.max(Y[:9]):
	df_sorted = df.sort_values(by=['Prediction', 'Catalyst'], ascending=[False, True])
	preliminary = df_sorted.iloc[0:5]
	df_sorted2 = preliminary.sort_values(by=['Variance', 'Catalyst'], ascending=[True, True])
	toPerform = df_sorted2.iloc[0]

else:
	df_sorted = df.sort_values(by=['Prediction', 'Catalyst'], ascending=[False, True])
	preliminary = df_sorted.iloc[0:10]
	df_sorted2 = preliminary.sort_values(by=['Variance', 'Catalyst'], ascending=[False, True])
	toPerform = df_sorted2.iloc[0]

#save data
feat_imp.to_csv(os.path.join('output_files', 'feature_importances.txt'), sep= ',')
best_params.to_csv(os.path.join('output_files', 'best_parameters.txt'), sep= ',')
toPerform.to_csv(os.path.join('output_files', 'selected_reaction.txt'), sep = ',')
df_sorted.to_csv(os.path.join('output_files', 'predictions.txt'), sep = ',')
filename3 = os.path.join('output_files', 'random_forest_model_grid.sav')
dump(grid, filename3)

print('Have a good one, mate!')



