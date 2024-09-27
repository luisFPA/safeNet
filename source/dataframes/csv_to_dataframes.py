#! python3
#r: openpyxl, psycopg2, Ipython

#THIS IS NECESSARY TO LOAD PANDAS otherwise it bugs
import locale
locale.setlocale(locale.LC_ALL, 'en_US')

#############################################################

import pandas as pd
import matplotlib as plt
import geopandas as gpd



dataFrame = pd.read_csv(CSV)