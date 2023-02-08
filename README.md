
## Running the pipeline

The pipeline creates training data from AIS data source and uses this data to optimize the best hyperparameters for the DBSCAN algorithm. 
The whole pipeline can be executed with command

```python main.py```

## Preprocessing

The preprocessing phase fetches AIS data from database and does filtering on the it. This step also calculates the center coordinates from this data as training data for the optimizer phase and saves these to the database. 
The preprocessing phase is done by the Preprocessing object, that can be run individually by importing the class object and initializing it. To run the preprocessing, go to a python shell and import the object with code:

```from preprocessing import Preprocessing```

Then execute ```Preprocessing()``` to run the preprocessing steps. 


## Optimizer

The optimizer phase finds the optimal parameters for a given algorithm using genetic algorithm as the optimizer. The results are saved to the optimizer object and the corresponding polygons are saved to the database. 
To run the optimizer step individually, go to a python shell and and import the object with code:
```from optimizer import Optimizer```

Then execute ```opt = Optimizer()``` to generate an object and
```opt.optimize(2015)``` to start the optimization step

## Config

The credentials to access monetdb are in the ```config.py``` file. 