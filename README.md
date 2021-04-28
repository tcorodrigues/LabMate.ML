# LabMate.ML 

LabMate.ML was designed to help identifying optimized conditions for chemical reactions.

The environment can be built from the `environment.yml` file:

```
conda env create -f environment.yml
```




## initializer.py

This script creates two text files that are required to run LabMate.AI:

1) `all_combos.txt` - the file contains all possible combinations of reaction conditions.
2) `train_data.txt` - the file contains 10 random reaction sampled from all possible combinations.


The script requires minor editing and instructions are available once the script is opened with a text editor.
To run the script, open a terminal in your destiny folder and type:

```
python initializer.py
```

After performing the reactions, add a column in the end of the `train_data.txt` file mentioning the reaction yield/conversion or similar (sample file available)


## optimizer.py


This script implements a routine to search for the next best experiment to be carried out.


To run LabMate.ML, open a terminal and type:

```
python optimizer.py
```

in the folder where the Python script, the `train_data.txt` and the `all_combos.txt` files are located. The columns in the txt files must be tab separated.
The grid search routine will run in 6 CPUs (`n_jobs=6` as in line 56), so make sure there are enough computational resources available.



Train data txt file:
- Fist column is the reaction identifier
- Last column is the reaction yield/conversion
- Columns in the middle correspond to the descriptor set
- File must be named `train_data.txt`, otherwise it will not be recognised by the script.


Search space txt file:
- Fist column is the reaction identifier 
- Following columns correspond to the descriptor set
- File must be named `all_combos.txt`, otherwise it will not be recognised by the script.


Output files:
- `best_score.txt` - saves the negative mean absolute error value (lower absolute value is better)
- `feature_importances.txt` - importance (given in the range of 0-1) for each descriptor, according to the random forest algorithm
- `selected_reaction.txt` - this is the next best experiment, as suggested by LabMate.AI
- `predictions.txt` - predictions for all possible reactions
- `random_forest_model_grid.sav` - saves the model






