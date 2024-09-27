#! python3
#r: openpyxl, psycopg2, Ipython

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Assume 'dataframe' is a pandas DataFrame input with a single column

# Step 1: Extract the column name from the DataFrame
column = dataframe.columns[0]

# Step 2: Perform scaling using StandardScaler
sc = StandardScaler()
output = sc.fit_transform(dataframe[[column]])

# Step 3: Create a new DataFrame with the scaled data and the original column name
scaled_df = pd.DataFrame(output, columns=[column])

