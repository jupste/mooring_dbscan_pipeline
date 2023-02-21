import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, euclidean_distances
import pygad
import pymonetdb
import tqdm
import config
from shapely import geometry
import utm


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
        self.pbar = tqdm.tqdm(total=6)
        self.algorithm = algorithm
        self.connection = self._database_connection()
        self._set_data()
        self._set_train()
        self.score_metric = score_metric
        self.search_params = search_params
        self.best_estimator = None
        self.best_params = None
        self.best_score = None
        self.poly = None
        self.stored_estimators = {}

    def _database_connection(self):
        self.pbar.set_description("[Creating database connection]")
        self.pbar.update(1)
        connection = pymonetdb.connect(username=config.MONETDB_USERNAME, password=config.MONETDB_PASSWORD,
                                       hostname=config.DB_URL, database=config.DB_NAME)
        return connection

    def set_polygons(self, poly):
        self.poly = poly
    def _fetch_data(self):
        self.pbar.set_description("[Fetching data from database]")
        self.pbar.update(1)
        cursor = self.connection.cursor()
        cursor.arraysize = 10000
        _ = cursor.execute(f'SELECT lat, lon, lat_utm, lon_utm, heading FROM train')
        self.connection.commit()
        return pd.DataFrame(cursor.fetchall(), columns=['lat', 'lon', 'lat_utm', 'lon_utm', 'heading'])

    def _set_data(self):
        data = self._fetch_data()
        self.data = self._set_utm(data)

    def _set_train(self):
        self.pbar.set_description("[Parsing training data]")
        self.pbar.update(1)
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
                if self._absolute_angle_difference(directions[i], directions[j]) > 15:
                    dir_matrix[i][j] = 10000
        return dir_matrix

    def _set_utm(self, data):
        data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.lon, data.lat), crs=4326)
        utm_zone = utm.from_latlon(*data.iloc[0][['lat', 'lon']].values)
        if utm_zone[3] > 'N':
            epsg = '326'
        else:
            epsg = '327'
        epsg = epsg + str(utm_zone[2])
        return data.to_crs(f'epsg:{epsg}')

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

    def make_polygons(self):
        '''
        Create convex hull polygons from points in clusters

        Parameters
        ----------
        clusters : DataFrame
            point dataframe with cluster membership
        Returns
        clusters : DataFrame
            dataframe with shapely polygons covering the area of clusters
        '''
        self.pbar.set_description("[Creating polygons]")
        clusters = self.data.copy()
        clusters['cluster'] = self.best_estimator.labels_
        clusters.sort_values(by=['cluster'], ascending=[True], inplace=True)
        clusters.reset_index(drop=True, inplace=True)
        clusters['geometry'] = [geometry.Point(xy) for xy in zip(clusters['lon'], clusters['lat'])]
        poly_clusters = gpd.GeoDataFrame()
        gb = clusters.groupby('cluster')
        for y in gb.groups:
            df0 = gb.get_group(y).copy()
            point_collection = geometry.MultiPoint(list(df0['geometry']))
            convex_hull_polygon = point_collection.convex_hull
            poly_clusters = pd.concat(
                [poly_clusters, pd.DataFrame(data={'cluster_id': [y], 'geometry': [convex_hull_polygon]})])
        poly_clusters.reset_index(drop=True, inplace=True)
        poly_clusters.crs = 'epsg:4326'
        poly_clusters['size'] = poly_clusters.cluster_id.map(clusters.cluster.value_counts())
        return gpd.GeoDataFrame(poly_clusters, crs='epsg:4326')

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
        self._optimize(**alg_kwargs)
        self.stored_estimators[year] = self.best_estimator
        #self._store_polygons()
        self.poly = self.make_polygons()

    def store_polygons(self):
        poly = self.poly.copy()
        poly = poly.to_wkt()
        poly.duration = poly.duration.astype('str')
        self.pbar.update(1)
        self.pbar.set_description("[Storing polygon data to database]")
        cursor = self.connection.cursor()
        _ = cursor.execute("DROP TABLE IF EXISTS polygons")
        self.connection.commit()
        _ = cursor.execute(
            "CREATE TABLE IF NOT EXISTS polygons (cluster_id INT, geometry string, size INT, arrival_time INT, "
            "duration string)")
        self.connection.commit()
        cols = ",".join([str(i) for i in poly.columns.tolist()])
        print(cols)
        for i, row in poly.iterrows():
            sql = f'INSERT INTO polygons ({cols}) VALUES {tuple(row)}'
            cursor.execute(sql)
            self.connection.commit()
        self.pbar.update(1)


    def predict(self, train, year):
        # TODO: Work in  progress
        if year in self.stored_estimators:
            return self.stored_estimators[year]
        else:
            self.optimize(train, year)
            return self.best_estimator

    def _optimize(self, **alg_kwargs):
        '''
        Optimize using genetic algorithm

        Parameters
        ----------
        train: numpy.array
            training data in utm form
        '''
        self.pbar.set_description("[Optimization step]")
        gene_space = list(self.search_params.values())
        params = list(self.search_params.keys())
        num_genes = 2
        num_generations = 200
        with tqdm.tqdm(total=num_generations, desc="[Optimizing with genetic algorithm]") as pbar:
            ga_instance = pygad.GA(num_generations=num_generations,
                                   sol_per_pop=10,
                                   num_parents_mating=5,
                                   keep_parents=2,
                                   num_genes=num_genes,
                                   gene_space=gene_space,
                                   init_range_high=2,
                                   parent_selection_type='rank',
                                   init_range_low=2,
                                   mutation_probability=0.5,
                                   stop_criteria='saturate_50',
                                   fitness_func=self._fitness_func(),
                                   suppress_warnings=True,
                                   on_generation=lambda _: pbar.update(1))
            ga_instance.run()
        solutions = ga_instance.best_solution()
        self.best_score = solutions[1]
        params = {params[0]: int(solutions[0][0]), params[1]: int(solutions[0][1])}
        self.best_params = params
        self.best_estimator = self.algorithm(**params, metric='precomputed').fit(self.train)

    def _fitness_func(self):
        train = self.train
        data = self.data.copy()

        def fitness_function(solution, solution_idx):
            model = self.algorithm(min_samples=int(solution[0]), eps=int(solution[1]), metric='precomputed')
            model.fit(train)
            try:
                score = silhouette_score(self.train, model.labels_, metric='precomputed')
            except:
                return -99
            if np.isnan(score):
                return -99
            data['cluster'] = model.labels_
            ratio = self._get_length_width_ratio(data)
            return score * ratio

        return fitness_function

    def __del__(self):
        self.connection.close()
        self.pbar.set_description("[Closing database connection.. Done!]")
        self.pbar.update(1)


if __name__ == '__main__':
    pass
