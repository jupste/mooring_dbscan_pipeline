import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, euclidean_distances
import pygad
import pymonetdb
import tqdm
import config
from shapely import geometry

def database_connection():
    connection = pymonetdb.connect(username=config.MONETDB_USERNAME, password=config.MONETDB_PASSWORD,
                                   hostname=config.DB_URL, database=config.DB_NAME)
    return connection


def fetch_data(connection):
    cursor = connection.cursor()
    cursor.arraysize = 10000
    _ = cursor.execute(f'SELECT lat_utm, lon_utm, heading FROM train')
    connection.commit()
    return pd.DataFrame(cursor.fetchall(), columns=['lat_utm', 'lon_utm', 'heading'])


class Optimizer:
    def __init__(self, algorithm=DBSCAN, score_metric=silhouette_score,
                 search_params={'min_samples': list(range(4, 100)), 'eps': list(range(5, 500))}):
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
        self._set_data()
        self._set_train()
        self.score_metric = score_metric
        self.search_params = search_params
        self.best_estimator = None
        self.best_params = None
        self.best_score = None
        self.stored_estimators = {}

    def _set_data(self):
        data = fetch_data(database_connection())
        self.data = data.values

    def _set_train(self):
        data = self.data
        penalties = self._heading_penalty_matrix(data.heading.values)
        distances = euclidean_distances(data[['lat_utm', 'lon_utm']].values)
        self.train = penalties + distances

    def _absolute_angle_difference(self, target, source):
        a = target - source
        a = np.abs((a + 180) % 360 - 180)
        b = target - source - 180
        b = np.abs((b + 180) % 360 - 180)
        return min(a, b)

    def _heading_penalty_matrix(self, directions):
        dir_matrix = np.zeros([len(directions), len(directions)])
        for i in range(len(directions)):
            for j in range(len(directions)):
                if self._absolute_angle_difference(directions[i], directions[j]) > 40:
                    dir_matrix[i][j] = 10000
        return dir_matrix

    def _get_length_width_ratio(self, df):
        clusters = df.copy()
        clusters.sort_values(by=['cluster'], ascending=[True], inplace=True)
        clusters.reset_index(drop=True, inplace=True)
        clusters = clusters[clusters.cluster != -1]
        gb = clusters.groupby('cluster')
        ratios = []
        for y in gb.groups:
            df0 = gb.get_group(y).copy()
            point_collection = geometry.MultiPoint(list(df0['geometry']))
            convex_hull_polygon = point_collection.convex_hull
            if not isinstance(convex_hull_polygon, geometry.Polygon):
                ratios.append(0)
                continue
            box = convex_hull_polygon.minimum_rotated_rectangle
            x, y = box.exterior.coords.xy
            edge_length = (geometry.Point(x[0], y[0]).distance(geometry.Point(x[1], y[1])),
                           geometry.Point(x[1], y[1]).distance(geometry.Point(x[2], y[2])))
            length = max(edge_length)
            width = min(edge_length)
            if width < 1:
                width = 1
            weight = len(df0) / len(clusters)
            ratios.append(weight * (length / width))
        return np.sum(ratios)

    def optimize(self, year, **alg_kwargs):
        '''
        Optimizes the parameters using genetic algorithm. Stores the results into stored_estimator dictionary.

        Parameters
        ----------
        train: numpy.array
            training data in utm form
        year: int
            year the AIS data was collected        

        '''
        self.__optimize_ga(self.train, **alg_kwargs)
        self.stored_estimators[year] = self.best_estimator

    def predict(self, train, year):
        # TODO: Work in  progress
        if year in self.stored_estimators:
            return self.stored_estimators[year]
        else:
            self.optimize(train, year)
            return self.best_estimator

    def _optimize(self, train, **alg_kwargs):
        '''
        Optimize using genetic algorithm

        Parameters
        ----------
        train: numpy.array
            training data in utm form
        '''

        def _fitness_func(solution, solution_idx):

            train = self.train
            data = self.data.copy()

            def fitness_function(solution, solution_idx):
                model = self.algorithm(**{params[0]: int(solution[0]), params[1]: int(solution[1])},
                                       metric='precomputed')
                model.fit(train)
                try:
                    score = silhouette_score(self.train, model.labels_, metric='precomputed')
                except:
                    return -99
                if np.isnan(score):
                    return -99
                data['cluster'] = model.labels_
                ratio = self._get_length_width_ratio(data)
                return ratio * score
            return fitness_function


        gene_space = list(self.search_params.values())
        params = list(self.search_params.keys())
        num_genes = 2
        num_generations = 200
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
                                   fitness_func=_fitness_func,
                                   suppress_warnings=True,
                                   on_generation=lambda _: pbar.update(1))
            ga_instance.run()
        solutions = ga_instance.best_solution()
        self.best_score = solutions[1]
        params = {params[0]: solutions[0][0], params[1]: solutions[0][1]}
        self.best_params = params
        self.best_estimator = self.algorithm(**params).fit(self.train)


if __name__ == '__main__':
    opt = Optimizer()
    opt.optimize(2016)
