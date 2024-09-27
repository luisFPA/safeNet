#! python3
#r: openpyxl, psycopg2, Ipython

#THIS IS NECESSARY TO LOAD PANDAS otherwise it bugs
import locale
locale.setlocale(locale.LC_ALL, 'en_US')

#############################################################

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point



# Read CSV file
df = pd.read_csv(csv_file)



# Create point geometry based on coordinates
geometry = [Point(xy) for xy in zip(df[longitude_column_name], df[latitude_column_name])]
gdf = gpd.GeoDataFrame(df, geometry=geometry)

# Establish reference system (CRS), WGS84 for example
gdf.set_crs(epsg=4326, inplace=True)

# Display dataframe
GeoDataFrame = gdf

# Save as Shapefile
gdf.to_file(shp_file)

# print(GeoDataFrame.crs)


