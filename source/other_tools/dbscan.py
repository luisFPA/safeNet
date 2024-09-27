#! python3
#r: openpyxl, psycopg2, Ipython

import rhinoscriptsyntax as rs
import Rhino.Geometry as rg
from sklearn.cluster import DBSCAN
from Grasshopper import DataTree
from Grasshopper.Kernel.Data import GH_Path

# Inputs: 
# points: List of point GUIDs
# eps: Maximum distance between two points for them to be considered in the same neighborhood
# min_samples: The minimum number of points required to form a dense region

# Step 1: Extract coordinates from GUIDs
point_coordinates = []
for guid in points:
    point = rs.coerce3dpoint(guid)
    if point:
        point_coordinates.append([point.X, point.Y, point.Z])

# Step 2: Convert min_samples to integer
min_samples = int(min_samples)

# Step 3: Perform DBSCAN clustering
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
dbscan.fit(point_coordinates)
labels = dbscan.labels_

# Step 4: Replace outlier labels (-1) with 0
labels = [0 if label == -1 else label for label in labels]

# Step 5: Organize points into a data tree based on cluster labels
clustered_points_tree = DataTree[object]()
unique_labels = set(labels)

# Print unique labels for debugging
print("Unique labels:", unique_labels)

for label in unique_labels:
    # Create a path for each cluster label
    path = GH_Path(int(label))
    for idx, point_label in enumerate(labels):
        if point_label == label:
            clustered_points_tree.Add(points[idx], path)

# Create a list of clusters
point_clusters = labels

# Output
clusters = clustered_points_tree
cluster_num = point_clusters
