import numpy as np
import pandas as pd
import geopandas as gpd
import folium
import time
import utm
import googlemaps
from shapely.ops import nearest_points
from shapely.geometry import LineString, Point, Polygon, MultiPolygon
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import config as cfg


class Quays:
    def __init__(self):
        self.water_shape = gpd.read_file(cfg.WATER_SHAPE)
        #self.api_connector = googlemaps.Client(key=cfg.MAPS_API_KEY)
        self.moor_polys = gpd.read_file(cfg.MOOR_POLYS)
        self.crs = self.get_utm()
        self.geo_places = None
        self.normalized_places = None
        self.quays = None
        self.geo_places = pd.read_csv('../trajectory_extraction/lb_terminals.csv')
        self.geo_places.geometry = gpd.GeoSeries.from_wkt(self.geo_places.geometry)
        self.geo_places = gpd.GeoDataFrame(self.geo_places, geometry = self.geo_places.geometry, crs = 4326)
        self.normalize_location()
        # self.find_places_of_interest()
        self.combine_polys_with_points()
        self.capture_stray_polygons()
        self.remove_intersecting_polys()
        self.calculate_angle()
        self.find_quays()

    def find_places_of_interest(self):
        response = self.api_connector.places(query = cfg.SEARCH_QUERY, location = cfg.LOCATION_COORDS, radius = cfg.SEARCH_RADIUS)
        df = gpd.GeoDataFrame(response['results'], crs = 4326)
        while True:
            try:
                response["next_page_token"]
            except:
                break
            response = self.api_connector.places(query = cfg.SEARCH_QUERY, location = cfg.LOCATION_COORDS, radius = cfg.SEARCH_RADIUS, page_token=response['next_page_token'])
            time.sleep(5)
            df = pd.concat([df, gpd.GeoDataFrame(response['results'])], ignore_index=True)
        df['lat'] = df['geometry'].apply(lambda x: list(x['location'].values())[0])
        df['lon'] = df['geometry'].apply(lambda x: list(x['location'].values())[1])
        df.geometry = gpd.GeoSeries.from_wkt(df.geometry)
        df = gpd.GeoDataFrame(df, geometry=df.geometry, crs = 4326)
        df['geometry'] = gpd.points_from_xy(df.lon, df.lat)
        self.geo_places = df
        self.normalize_location()
    def normalize_location(self):
        df = self.geo_places[self.geo_places.to_crs(self.crs).geometry.apply(lambda x: x.distance(self.water_shape.to_crs(self.crs).iloc[0].geometry)) < 400]
        df.geometry = df.geometry.apply(lambda x: nearest_points(x, self.water_shape.iloc[0].geometry)[1])
        self.normalized_places = df
    def get_utm(self):
        utm_zone = utm.from_latlon(*cfg.LOCATION_COORDS)
        if utm_zone[3] > 'N':
            epsg = '326'
        else:
            epsg = '327'
        epsg = epsg + str(utm_zone[2])
        return int(epsg)

    def combine_polys_with_points(self):
        self.moor_polys.geometry = self.moor_polys.convex_hull
        self.moor_polys['closest'] = self.normalized_places.sindex.nearest(self.moor_polys.geometry)[1]
        coords = [(x, y) for x, y in zip(self.moor_polys['geometry'].centroid.x, self.moor_polys['geometry'].centroid.y)]
        linked = self.normalized_places.iloc[list(self.moor_polys.closest.unique())]
        seeds = [(x, y) for x, y in zip(linked['geometry'].x, linked['geometry'].y)]
        model = KMeans(n_clusters=len(seeds), init=seeds)
        model.fit(coords)
        linked.geometry = [Point(x, y) for x, y in model.cluster_centers_]
        linked.reset_index(drop = True, inplace = True)
        self.moor_polys['closest'] = linked.sindex.nearest(self.moor_polys.geometry)[1]
        self.normalized_places = linked

    def _small_intersection(self, x):
        for geo in self.moor_polys.geometry.values:
            if geo.intersects(x):
                if geo.intersection(x).area > x.area * 0.25 and geo.area > x.area:
                    return False
        return True

    def _segments(self, curve):
        return list(map(LineString, zip(curve.coords[:-1], curve.coords[1:])))
    def get_longest_side_azimuth(self, poly):
        seg = self._segments(poly.boundary)
        line = seg[np.argmax([x.length for x in seg])]
        coords = [*line.coords]
        angle = np.arctan2(coords[1][0] - coords[0][0], coords[1][1] - coords[0][1])
        deg = np.degrees(angle) if angle >= 0 else np.degrees(angle) + 360
        return deg % 180

    def remove_intersecting_polys(self):
        self.moor_polys = self.moor_polys[self.moor_polys.geometry.apply(lambda x: self._small_intersection(x))]

    def calculate_angle(self):
        self.moor_polys['angle'] = self.moor_polys.geometry.apply(lambda x: self.get_longest_side_azimuth(x))

    def capture_stray_polygons(self):
        for i, row in self.moor_polys.iterrows():
            s = self.moor_polys.drop(index=i)
            nearest_poly = s.sindex.nearest(row.geometry)
            index = nearest_poly[1][0]
            self.moor_polys.at[i, 'closest'] = s.iloc[index]['closest']

    def _combine_collinear_points(self, rows):
        groups = []
        df = rows.copy()
        while len(df) > 0:
            median_angle = df.angle.quantile(interpolation='lower')
            med_row = df[df.angle == median_angle]
            median_point = med_row.iloc[0].center_point
            group = df[df.apply(lambda x: self._absolute_angle_difference(median_angle, x.angle) < 10 and
                                          self._check_collinearity(x.center_point, x.proj_point, median_point), axis=1)]
            groups.append(group)
            df.drop(index=group.index, inplace=True)
        return groups

    def _absolute_angle_difference(self, target, source):
        a = target - source
        a = np.abs((a + 180) % 360 - 180)
        b = target - source - 180
        b = np.abs((b + 180) % 360 - 180)
        return min(a, b)
    def find_quays(self):
        gdf_temp = self.moor_polys.copy()
        gdf_temp.to_crs(self.get_utm(), inplace = True)
        gdf_temp.geometry = gdf_temp.centroid
        proj_points = gdf_temp.apply(lambda x: self._create_auxilliary_point(x.geometry, x.angle), axis = 1)
        proj_points = proj_points.set_crs(self.get_utm())
        proj_points = proj_points.to_crs(4326)
        self.moor_polys['proj_point'] = proj_points
        self.moor_polys['center_point'] = self.moor_polys.centroid
        q = self.moor_polys.groupby('closest').apply(lambda x: self._combine_collinear_points(x))
        quay_polys = []
        closest = []
        size = []
        for quay in q:
            for row in quay:
                quay_polys.append(MultiPolygon([*row.geometry]).convex_hull)
                closest.append(row.closest.mean())
                size.append(len(row))
        self.quays = gpd.GeoDataFrame(data={'closest': closest, 'size': size}, geometry=quay_polys,
                                        columns=['closest', 'size'], crs=4326)
        self.quays.closest.map(self.normalized_places['name'])

    def _check_collinearity(self, center, projection, point):
        a = np.array([[center.x,center.y,1],[projection.x,projection.y,1],[point.x,point.y,1]])
        area = 0.5*np.linalg.det(a)
        distance = center.distance(point)
        if distance == 0:
            return True
        return abs(area)/distance<10e-5
    def _create_auxilliary_point(self, point, angle, d=100):
        alpha = np.radians(angle)
        xx = point.x + (d * np.sin(alpha))
        yy = point.y + (d * np.cos(alpha))
        return Point([xx, yy])