#! python3
#r: openpyxl, psycopg2, Ipython

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Assume 'dataframe' is a pandas DataFrame input
# and 'column' is the column name provided as a string

# Step 1: Perform scaling using StandardScaler
sc = StandardScaler()
output = sc.fit_transform(dataframe[[column]])

# Step 2: Convert the output to a list that Grasshopper can handle
# Convert the output (which is a NumPy array) to a Python list
scaled_list = output.flatten().tolist()

# Output: scaled_list can be connected to a panel or used in Grasshopper
scaled = scaled_list
