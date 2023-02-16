from operator import ge
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import geopandas as gpd
import pymonetdb
import config
from datetime import timedelta
import utm
import tqdm
from scipy.stats import circmean


# TODO: placeholder method before database is established
def filter_ship_types(df):
    static = pd.read_csv('static_data.csv', dtype={'sourcemmsi' : 'int'})
    static = static.rename(columns={'sourcemmsi': 'mmsi'})
    vessel_types = static.groupby('mmsi').shiptype.max()
    df['vessel_type'] = df.mmsi.astype('int').map(vessel_types)
    return df[df.vessel_type.isin(list(range(70, 90)))]


class Preprocessing:
    def __init__(self):
        self.pbar = tqdm.tqdm(total=12)
        self.connection = self._database_connection()
        self.data = self._fetch_data(2015, "Brest")
        self.preprocess_data()
        self.calculate_groups()
        self.agg_data = self.calculate_centers()
        self._change_coordinates_to_utm()
        self._store_train_data()
    def _database_connection(self):
        '''
        Setup database connection with credentials configured in config.py
        '''
        self.pbar.update(1)
        self.pbar.set_description("[Creating database connection]")
        connection = pymonetdb.connect(username=config.MONETDB_USERNAME, password=config.MONETDB_PASSWORD,
                                       hostname=config.DB_URL, database=config.DB_NAME)
        return connection

    def _fetch_data(self, year, port):
        '''
        Fetch data from database based on year and area

        Parameters
        ----------
        year: int
            timeframe of the data
        port: string
            name of port


        Returns
        -------
        df: DataFrame
            AIS data in pandas dataframe format

        '''
        # TODO: fetch from MonetDB data that corresponds to the year and port parameters
        self.pbar.update(3)
        self.pbar.set_description("[Querying data from database]")
        cursor = self.connection.cursor()
        cursor.arraysize = 5_000_000
        # Create a mask for the bounding box?
        # Read from WPI to get the coordinates for the port?
        _ = cursor.execute(
            f'SELECT {config.COLUMN_NAMES["mmsi"]}, {config.COLUMN_NAMES["status"]}, {config.COLUMN_NAMES["speed"]}, {config.COLUMN_NAMES["heading"]},'
            f'{config.COLUMN_NAMES["lon"]}, {config.COLUMN_NAMES["lat"]}, {config.COLUMN_NAMES["time"]}, {config.COLUMN_NAMES["shiptype"]} FROM brest_dynamic WHERE shiptype BETWEEN 70 AND 89')
        self.connection.commit()
        self.pbar.update(1)
        df = pd.DataFrame(cursor.fetchall(),
                     columns=['mmsi', 'status', 'speed', 'heading', 'lon', 'lat', 'time', 'shiptype'])

        df = df.astype(dtype={'mmsi': 'category', 'status': 'category', 'heading': 'float', 'lon': 'float', 'lat': 'float', 'shiptype': 'category'})
        return df

    def preprocess_data(self):
        '''
        Preprocess the dataframe.
        The following steps are execued:

        drop mmsi/id duplicates
        drop rows that are moored and have high speed (>1 knots)
        drop all rows from ships that have less than 1000 messages
        change id column to categorical


        Parameters
        ----------
        df : DataFrame
            raw AIS dataframe


        Returns
        -------
        df : DataFrame
            preprocessed AIS dataframe
        '''
        self.pbar.set_description('[Processing data]')
        data = gpd.GeoDataFrame(self.data, geometry=gpd.points_from_xy(self.data.lon, self.data.lat), crs='epsg:4326')
        data.drop_duplicates(['mmsi', 'time'], inplace=True)
        data.drop(data[(data.speed > 1) & (data.status == 5)].index, inplace=True)
        # Drop rows where the heading is more than 360
        data.drop(data[(data.heading > 360)].index, inplace=True)
        # Drop ships with less than 1000 messages
        data['date'] = pd.to_datetime(data.time, unit='s')
        mmsis = (data.groupby('mmsi').size() > 100)
        mmsis = mmsis[mmsis == True]
        data = data[data.mmsi.isin(list(mmsis[mmsis == True].index))]
        data.status.fillna(15, inplace=True)
        # TODO: remove after static data is included to db
        #data = filter_ship_types(data)
        data = data[data.shiptype.isin(list(range(70, 90)))]
        self.data = data.copy()
        self._filter_by_area()
        self.pbar.update(1)

    def _filter_by_area(self):
        '''
        Filter the dataset by a bounding box
        '''
        # TODO: currently using hardcoded bounding box for Brest harbor. Find a way to capture the area in a dynamic manner
        minx, miny, maxx, maxy = -4.55230359, 48.33937308, -4.42895155, 48.38951694
        self.data = self.data.cx[minx:maxx, miny:maxy]

    def calculate_groups(self, group_name='status', column_name='berth_num'):
        '''
        Detect continuous time periods where a ship has the same value in a column. Defaults to navigational_status. A new number is given when either the value
        or the ship mmsi changes in the dataframe

        Parameters
        ----------
        df : DataFrame
            AIS dataframe, sorted by mmsi, t
        group_name : string (default 'status')
            name of group
        column_name : string (default 'berth_num)
            name of new column added to the dataframe

        Returns
        -------
        df : DataFrame
            dataframe with berth column
        '''
        self.data.sort_values(['mmsi', 'time'], inplace=True)
        self.data.loc[:, column_name] = ((self.data[group_name].diff() != 0) | (self.data.mmsi.ne(self.data.mmsi.shift())) | (
                self.data.date.diff() > timedelta(hours=2))).cumsum()
        self.pbar.update(1)

    def calculate_centers(self, moor=5):
        '''
        Calculate center points from each individual mooring event


        Parameters
        ----------
        df : DataFrame
            pandas dataframe that contains the AIS messages
        moor : int or string
            the status code for mooring. Defaults to 5

        Returns
        center_df : new dataframe that has the median and standard deviation of coordinates of each mooring visit as well as the duration of the visit

        -------'''
        # df.sort_values(['mmsi', 'time'], inplace=True)
        self.pbar.set_description("[Creating training data]")
        self.pbar.update(1)
        berth_visits = self.data[self.data.status == moor].groupby('berth_num')
        lon = berth_visits.lon.apply(pd.Series.median)
        lat = berth_visits.lat.apply(pd.Series.median)
        heading = berth_visits.heading.apply(lambda x: np.degrees(circmean(np.radians(x))))
        time = berth_visits.date.max() - berth_visits.date.min()
        centers_df = pd.DataFrame(list(zip(lon.values, lat.values, time.values, heading.values, list(lon.index))),
                                  columns=['lon', 'lat', 'time', 'heading', 'berth_num'])
        self.pbar.update(1)
        return centers_df

    def _change_coordinates_to_utm(self):
        '''
        Convert the coordinates to Universal Traverse Mercaptor coordinates, which allow the calculating distances to meters without haversine calculation.
        Method detects UTM zone and converts the coordinates accordingly

        Parameters
        ----------
        df: DataFrame
            AIS dataframe

        Returns
        -------
        df: DataFrame
            AIS dataframe with added UTM coordinate columns
        '''
        gdf = gpd.GeoDataFrame(self.agg_data, geometry=gpd.points_from_xy(self.agg_data.lon, self.agg_data.lat),
                               crs='epsg:4326')
        utm_zone = utm.from_latlon(*gdf.iloc[0][['lat', 'lon']].values)
        if utm_zone[3] > 'N':
            epsg = '326'
        else:
            epsg = '327'
        epsg = epsg + str(utm_zone[2])

        gdf.to_crs(f'epsg:{epsg}', inplace=True)
        self.agg_data['lon_utm'] = gdf.geometry.x
        self.agg_data['lat_utm'] = gdf.geometry.y

    def _store_train_data(self):
        '''
        Store training data to database

        Parameters
        ----------
        df: DataFrame
            training data for optimization
        '''
        self.pbar.update(1)
        self.pbar.set_description("[Storing train data to database]")
        cursor = self.connection.cursor()
        cursor.arraysize  = 1000
        _ = cursor.execute("DROP TABLE IF EXISTS train")
        self.connection.commit()
        _ = cursor.execute(
            "CREATE TABLE IF NOT EXISTS train (lat REAL, lon REAL, lat_utm REAL, lon_utm REAL, heading REAL)")
        self.connection.commit()
        data = self.agg_data[['lat', 'lon', 'lat_utm', 'lon_utm', 'heading']].copy()
        cols = ",".join([str(i) for i in data.columns.tolist()])
        for i, row in data.iterrows():
            sql = f'INSERT INTO train ({cols}) VALUES {tuple(row)}'
            cursor.execute(sql)
            self.connection.commit()
        self.pbar.update(1)
        self.connection.close()
        self.pbar.set_description("[Closing database connection.. Done!]")
        self.pbar.update(1)


if __name__ == '__main__':
    pass
