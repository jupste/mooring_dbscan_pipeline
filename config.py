MONETDB_USERNAME = 'monetdb'#'vesselai'
MONETDB_PASSWORD = 'monetdb'#'vesselai'
DB_URL = 'localhost' #'128.214.253.154'
DB_NAME = 'brest' #'ais_data'
COLUMN_NAMES = {'mmsi' : 'sourcemmsi', 'heading' : 'trueheading', 'speed' : 'speedoverground', 'status' : 'navigationalstatus', 'time' : 't', 'lon' : 'lon', 'lat' : 'lat'}
LOCATION_COORDS = (33.745, -118.244)
SEARCH_QUERY = ""
MAPS_API_KEY = ""

SEARCH_RADIUS = 1000
WATER_SHAPE = "../trajectory_extraction/water.json"
MOOR_POLYS = "../trajectory_extraction/berth_polygons.geojson"
