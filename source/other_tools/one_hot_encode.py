#! python3
#r: openpyxl, psycopg2, Ipython

import pandas as pd
from Grasshopper import DataTree
from Grasshopper.Kernel.Data import GH_Path

# Inputs:
# cluster_num: List of cluster indices where each index represents the cluster number

# Convert the list of cluster indices to a pandas DataFrame
df = pd.DataFrame({'cluster': cluster_labels})

# Perform one-hot encoding
one_hot = pd.get_dummies(df['cluster'])

# Initialize a new DataTree to hold the one-hot encoded results
one_hot_tree = DataTree[object]()

# Iterate over each cluster and add its one-hot encoded values to the DataTree
for i, column in enumerate(one_hot.columns):
    path = GH_Path(i)  # Create a new path for each cluster
    # Add the column values as a list to the corresponding path in the DataTree
    one_hot_tree.AddRange(one_hot[column].tolist(), path)

# Output the DataTree
output = one_hot_tree
