import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import circmean

class Statistics:
    def __init__(self, data, polygons):
        self._combine_polys_to_data(data, polygons)
        self.clusters = polygons
        self.clusters.index = self.clusters.cluster_id
        #self._calculate_max_dimensions()
        self._calculate_arrival_and_staying_time()

    def _combine_polys_to_data(self, data, polygons):
        '''
        Adds cluster membership to each row in an AIS dataframe

        Parameters
        ----------
        gdf : GeoDataFrame
            geodataframe containing AIS messages
        polygons : GeoDataFrame
            clusters to be added to the AIS data
        cluster_type : string
            type of cluster
        Returns
        clusters : GeoDataFrame
            AIS dataframe with cluster membership
        '''
        data = data[data.status==5]
        data['cluster'] = -1
        for _, cluster in polygons[polygons.cluster_id!=-1].iterrows():
            geom = cluster.geometry
            sindex = data.sindex
            possible_matches_index = list(sindex.intersection(geom.bounds))
            possible_matches = data.iloc[possible_matches_index]
            precise_matches = possible_matches[possible_matches.intersects(geom)]
            data.loc[precise_matches.index, 'cluster'] = cluster.cluster_id
        self.data = data

    def _calculate_max_dimensions(self):
        gb = self.data.groupby('cluster')
        self.clusters['max_width'] = gb.width.quantile(0.99)
        self.clusters['max_length'] = gb.length.quantile(0.99)
        self.clusters['max_draft'] = gb.draft.quantile(0.99)

    def _calculate_arrival_and_staying_time(self):
        c = self.data.copy()
        c.sort_values(['mmsi', 'date'], inplace=True)
        c['new_cluster'] = ((c.cluster != c.cluster.shift()) | (c.mmsi != c.mmsi.shift()))
        c['cluster_num'] = c.new_cluster.cumsum()
        gb = c.groupby('cluster_num')
        visits = pd.DataFrame.from_dict({'arrival': gb.date.min().apply(lambda x: x.hour), 'duration': gb.date.max()-gb.date.min(), 'cluster': gb.cluster.min()})
        vgb = visits.groupby('cluster')
        self.clusters['arrival_time'] = vgb.arrival.agg(circmean, high=24)
        self.clusters['duration'] = vgb.duration.mean()
