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

def database_connection():
    '''
    Setup database connection with credentials configured in config.py
    '''
    connection = pymonetdb.connect(username=config.MONETDB_USERNAME, password=config.MONETDB_PASSWORD, hostname=config.DB_URL, database=config.DB_NAME)
    return connection


def fetch_data(year, port):
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
    # Create a mask for the bounding box?
    # Read from WPI to get the coordinates for the port?
    _ = cursor.execute(f'SELECT {config.COLUMN_NAMES["mmsi"]}, {config.COLUMN_NAMES["status"]}, {config.COLUMN_NAMES["speed"]}, {config.COLUMN_NAMES["heading"]},'
       f'{config.COLUMN_NAMES["lon"]}, {config.COLUMN_NAMES["lat"]}, {config.COLUMN_NAMES["time"]} FROM brest_ais') # WHERE shiptype BETWEEN 70 AND 89')
    connection.commit()
    return pd.DataFrame(cursor.fetchall(), columns = ['mmsi', 'status', 'speed', 'heading', 'lon','lat', 'time'])

def preprocess_data(df):
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
    df = gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df.lon, df.lat), crs='epsg:4326')
    df.drop_duplicates(['mmsi', 'time'], inplace=True)
    df.drop(df[(df.speed>1) & (df.status==5)].index, inplace=True)
    # Drop rows where the heading is more than 360
    df.drop(df[(df.heading>360)].index, inplace=True)
    # Drop ships with less than 1000 messages
    df['date'] = pd.to_datetime(df.time, unit='s')
    mmsis = (df.groupby('mmsi').size()>200)
    mmsis = mmsis[mmsis==True]
    df = df[df.mmsi.isin(list(mmsis[mmsis==True].index))]
    df.status.fillna(15,inplace=True)
    df = filter_by_area(df)
    return df


def filter_by_area(df):
    '''
    Filter the dataset by a bounding box
    '''
    #TODO: currently using hardcoded bounding box for Brest harbor. Find a way to capture the area in a dynamic manner
    minx, miny, maxx, maxy = -4.55230359, 48.33937308, -4.42895155, 48.38951694
    return df.cx[minx:maxx, miny:maxy]

def calculate_groups(df, group_name='status', column_name='berth_num'):
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
    df.sort_values(['mmsi','time'], inplace=True)
    df.loc[:,column_name] = ((df[group_name].diff()!=0) | (df.mmsi.ne(df.mmsi.shift())) | (df.date.diff()>timedelta(hours=2))).cumsum()
    return df

def calculate_centers(df, moor=5):
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
    df.sort_values(['mmsi', 'time'], inplace=True)
    berth_visits = df[df.status==moor].groupby('berth_num')
    lon = berth_visits.lon.apply(pd.Series.median)
    lat = berth_visits.lat.apply(pd.Series.median)
    heading = berth_visits.heading.apply(lambda x: np.degrees(circmean(np.radians(x))))
    time = berth_visits.date.max()-berth_visits.date.min()
    center_df = pd.DataFrame(list(zip(lon.values, lat.values, time.values, heading.values, list(lon.index))), columns=['lon', 'lat', 'time', 'heading', 'berth_num'])
    return center_df

def change_coordinates_to_utm(df):
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
    gdf = gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df.lon, df.lat), crs='epsg:4326')
    utm_zone = utm.from_latlon(*gdf.iloc[0][['lat', 'lon']].values)
    if utm_zone[3]>'N':
        epsg = '326'
    else:
        epsg = '327'
    epsg = epsg + str(utm_zone[2])
    gdf.to_crs('epsg:'+epsg, inplace=True)
    df['lon_utm'] = gdf.geometry.x
    df['lat_utm'] = gdf.geometry.y
    return df


def store_train_data(df):
    '''
    Store training data to database

    Parameters
    ----------
    df: DataFrame
        training data for optimization 
    '''
    _ = cursor.execute("DROP TABLE IF EXISTS train")
    connection.commit()
    _ = cursor.execute("CREATE TABLE IF NOT EXISTS train (lat REAL, lon REAL, lat_utm REAL, lon_utm REAL, heading REAL)")
    connection.commit()
    data = df[['lat', 'lon', 'lat_utm','lon_utm', 'heading']].copy()
    cols = ",".join([str(i) for i in data.columns.tolist()])
    for i,row in data.iterrows():
        sql = f'INSERT INTO train ({cols}) VALUES {tuple(row)}'
        cursor.execute(sql)
        connection.commit()


if __name__ == '__main__':
    pbar = tqdm.tqdm(total=12)
    pbar.update(1)
    pbar.set_description("[Creating database connection]")
    connection = database_connection()
    cursor = connection.cursor()
    cursor.arraysize = 10000
    pbar.update(3)
    pbar.set_description("[Querying data from database]")
    pbar.update(1)
    df = fetch_data(2015, "")
    pbar.set_description('[Processing data]')
    df = preprocess_data(df)
    pbar.update(1)
    df = calculate_groups(df)
    pbar.update(1)
    pbar.set_description("[Creating training data]")
    pbar.update(1)
    centers = calculate_centers(df)
    pbar.update(1)
    centers = change_coordinates_to_utm(centers)
    pbar.update(1)
    pbar.set_description("[Storing train data to database]")
    store_train_data(centers)
    pbar.update(1)
    pbar.set_description("[Closing database connection.. Done!]")
    pbar.update(1)
    connection.close()
