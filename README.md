# LabMate.ML 

LabMate.ML was designed to help identifying optimized conditions for chemical reactions.

## Installation
In order to use `LabMate.ML` you must first have [Anaconda](https://www.anaconda.com/products/individual) installed on your machine.
Once installed, use the below script to download the relevant dependencies and files.

```bash
$ git clone https://github.com/tcorodrigues/LabMate.ML.git
$ cd LabMate.ML
$ conda env create -f environment.yml
```

This will download and install the required dependencies for `LabMate.Ml` and store them in the conda environment `LabMateML`
To access this environment again later, simply type the below command in the terminal:

```
conda activate LabMateML
```

## initializer.py
The initializer script creates the two text files needed to run `LabeMate.AI`:
1. A file containint all possible combinations of reaction conditions (`all_combos.txt`)
2. Another containing a sample (`n >= 10`) of reaction conditions from the combinations (`train_data.txt`)

The script can be run using the below command in the terminal:
```bash
$ python initializer.py
```  

**After performing the reactions, add a column in the end of the `train_data.txt` file mentioning the reaction yield/conversion or similar (sample file available)**


### Customization
Different aspects of the initialisation process can be achieved by specifying the values in the command line as detailed below:

```bash
$ python initializer.py -- help

    usage: initializer.py [-h] [-i INIT_DIR] [-b BOUNDARY] [-s SEED] [-n N_SAMPLES]
    
    optional arguments:
      -h, --help,      show this help message and exit
      -i, --init_dir,  default='init_files',     dir to save files to.
      -b, --boundary,  default='Boundaries.yml', File containing boundary ranges.
      -s, --seed,      default=1,                Random seed for sampling.
      -n, --n_samples, default=10,               Number conditions to sample.
```

Hence, an initialisation using `20` random samples instead of `10` would be as below:
```bash
$ python initializer.py --n_samples 20
```

### Bondaries.yml
The Boundaries file allows for customisation of the different parameters of the reaction you wish to optimize
over, and can be edited as a regular text file using notepad or equivalent.
To include a new reaction condition to be optimised, simply add a keyword describing the condition and also list all values you wish that condition to be evaluated at.
For example, if we wished to list the stirring rate of the reaction at 100 and 200 rpm, then the below would be added:

```yaml
StirRate:
- 100
- 200
```


## optimizer.py


This script implements a routine to search for the next best experiment to be carried out.
It requires `initializer.py` to have been run first to generate the required files.

To run LabMate.ML, open a terminal and navigate to the directory containing the Python script, the `train_data.txt` and the `all_combos.txt` files.
Then use the below command:

```
python optimizer.py
```

Just as with `initializer.py` there are a number of optinal command line arguments which can be specified:
```bash
$ python optimizer.py -- help

    usage: optimizer.py [-h] [-o OUT_DIR] [-t TRAIN_FILE] [-i INIT_DIR] [-s SEED] [-m METRIC] [-c COMBOS_FILE] [-j JOBS]
    
    optional arguments:
      -h, --help,           show this help message and exit
      -o, --out_dir,        default='output_files',            dir to save files to.
      -t, --train_file,     default='train_data.txt',          Training data location.
      -i, --init_dir,       default='init_files',              dir to load files from.
      -s, --seed,           default=1,                         Random seed value.
      -m, --metric,         default='neg_mean_absolute_error', Metric for evaluatng hyperparameters.      
      -c, --combos_file,    default=all_combos.txt,            File containing all reaction combinations.
      -j, --jobs,           default=6,                         Number of parallel jobs when optimising hyperparameters.
```
Note : the grid search routine will use 6 CPUs unless otherwise specified so make sure there are enough computational resources available.

### File Requirements
The columns in the txt files must be tab separated.

#### train_data.txt
- Fist column is the reaction identifier
- Last column is the reaction yield/conversion
- Columns in the middle correspond to the descriptor set
- File must be named `train_data.txt`, otherwise it will not be recognised by the script.


#### all_combos.txt
- Fist column is the reaction identifier 
- Following columns correspond to the descriptor set
- File must be named `all_combos.txt`, otherwise it will not be recognised by the script.


### Output files:
- `best_score.txt` : saves the negative mean absolute error value (lower absolute value is better)
- `feature_importances.txt` : importance (given in the range of 0-1) for each descriptor, according to the random forest algorithm
- `selected_reaction.txt` : this is the next best experiment, as suggested by LabMate.AI
- `predictions.txt` : predictions for all possible reactions
- `random_forest_model_grid.sav` : saves the model
___





