#! python3
#r: openpyxl, psycopg2, Ipython

#THIS IS NECESSARY TO LOAD PANDAS otherwise it bugs
import locale
locale.setlocale(locale.LC_ALL, 'en_US')

import geopandas as gpd

GeoDataFrame = gpd.read_file(SHP)