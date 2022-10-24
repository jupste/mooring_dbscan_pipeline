## Preprocessing

The preprocessing phase fetches AIS data from database and does filtering on the it. This step also calculates the center coordinates from this data as training data for the optimizer phase and saves these to the database. 

Use command ```python preprocess.py``` to run the preprocessing steps. 


## Optimizer

The optimizer phase finds the optimal parameters for a given algorithm using genetic algorithm as the optimizer. The results are saved to the optimizer object.

Use command ```python optimizer.py``` to run the optimizer step.

## Config

The credentials to access monetdb are in the ```config.py``` file. 