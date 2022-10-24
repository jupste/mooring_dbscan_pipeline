import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import calinski_harabasz_score
import pygad
import pymonetdb
import tqdm
import config



def database_connection():
    connection = pymonetdb.connect(username=config.MONETDB_USERNAME, password=config.MONETDB_PASSWORD, hostname=config.DB_URL, database=config.DB_NAME)
    return connection

def fetch_data():
    _ = cursor.execute(f'SELECT lat_utm, lon_utm FROM train')
    connection.commit()
    return pd.DataFrame(cursor.fetchall(), columns = ['lat_utm','lon_utm'])

class Optimizer:
    def __init__(self, algorithm = DBSCAN, score_metric=calinski_harabasz_score, search_params = {'min_samples': list(range(5,100,5)), 'eps': list(range(5,800,25))}):
        '''
        Parameters
        ----------
        algorithm: sklearn.cluster method (default sklearn.cluster.DBSCAN)
            clustering algorithm that is optimized. Use either sklearn.cluster.DBSCAN, sklearn.cluster.optics or hdbscan.HDBSCAN
        score_metric: sklearn.metric (default sklearn.metric.calinski_harabasz_score)
            cluster validation metric that is used as the fitness function of the genetic algorithm
        search_params: dict (default {'min_samples': list(range(5,100,5)), 'eps': list(range(5,800,25))})
            search parameters for optimization. Format is variable name: search space i.e. min_samples([1,2,3,4]) would test values 1,2,3,4 for variable min_samples

        '''
        
        self.algorithm = algorithm
        self.score_metric = score_metric
        self.search_params = search_params
        self.best_estimator = None
        self.best_params = None
        self.best_score = None
        self.stored_estimators = {} 


    def optimize(self, train, year, **alg_kwargs):
        '''
        Optimizes the parameters using genetic algorithm. Stores the results into stored_estimator dictionary.

        Parameters
        ----------
        train: numpy.array
            training data in utm form
        year: int
            year the AIS data was collected        

        '''
        self.__optimize_ga(train, **alg_kwargs)
        self.stored_estimators[year] = self.best_estimator.fit(train)

    def predict(self, train, year):
        # TODO: Work in  progress
        if year in self.stored_estimators:
            return self.stored_estimators[year]
        else:
            self.optimize(train, year)
            return self.best_estimator
    

    def __optimize_cv(self, train, **alg_kwargs):
        # Old method that used gridsearch to optimize
        def __scorer(estimator, X):
            estimator.fit(X)
            try:
                return self.score_metric(X, estimator.labels_)
            except ValueError:
                return -1
        cv = self.optimizer(estimator=self.algorithm(**alg_kwargs), param_grid=self.search_params, 
                  scoring=__scorer, cv=[(slice(None), slice(None))], n_jobs=-1)
        cv.fit(train)
        self.best_estimator = cv.best_estimator_
        self.best_score = cv.best_score_
        self.best_params = cv.best_params_

    def __optimize_ga(self, train, **alg_kwargs):
        '''
        Optimize using genetic algorithm

        Parameters
        ----------
        train: numpy.array
            training data in utm form
        '''
        def __fitness_func(solution, solution_idx, **alg_kwargs):
            model = self.algorithm(**{params[0]:int(solution[0]), params[1]:int(solution[1])}, **alg_kwargs)
            model.fit(train)
            try:
                score = self.score_metric(train, model.labels_)
            except ValueError:
                return -1
            if np.isnan(score):
                return -1
            return score

        gene_space = list(self.search_params.values())
        params = list(self.search_params.keys())
        num_genes=2
        num_generations=200
        with tqdm.tqdm(total=num_generations, desc="[Optimizining with genetic algorithm]") as pbar:
            ga_instance = pygad.GA(num_generations=num_generations,
                        sol_per_pop=10,
                        num_parents_mating=5,
                        keep_parents=2,
                        num_genes=num_genes,
                        gene_space=gene_space,
                        init_range_high=2,
                        parent_selection_type='tournament',
                        init_range_low=2,
                        mutation_probability=0.5,
                        stop_criteria='saturate_20',
                        fitness_func=__fitness_func,
                        suppress_warnings=True, 
                        on_generation=lambda _: pbar.update(1))
            ga_instance.run()
        solutions = ga_instance.best_solution()
        self.best_score = solutions[1]
        params = {params[0]: solutions[0][0], params[1]:solutions[0][1]}
        self.best_params = params
        self.best_estimator = self.algorithm(**params)

if __name__=='__main__':
    connection = database_connection()
    cursor = connection.cursor()
    cursor.arraysize = 10000
    data = fetch_data()
    train = data.values
    opt = Optimizer()
    opt.optimize(train, 2016)
    print(opt.stored_estimators)
    print(opt.best_score)
    connection.close()