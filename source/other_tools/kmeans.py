#! python3
#r: openpyxl, psycopg2, Ipython

import rhinoscriptsyntax as rs
import Rhino.Geometry as rg
from sklearn.cluster import KMeans
from Grasshopper import DataTree
from Grasshopper.Kernel.Data import GH_Path

# Ensure 'points' is a list of GUIDs representing points in Rhino/Grasshopper
# Convert num_clusters to an integer
num_clusters = int(num_clusters)
seed = int(seed)

# Step 1: Extract coordinates from GUIDs
point_coordinates = []
for guid in points:
    point = rs.coerce3dpoint(guid)
    if point:
        point_coordinates.append([point.X, point.Y, point.Z])

# Step 2: Perform KMeans clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=seed, n_init="auto").fit(point_coordinates)
labels = kmeans.labels_

# Step 3: Organize points into a data tree based on cluster labels
clustered_points_tree = DataTree[object]()
for i in range(num_clusters):
    path = GH_Path(i)
    for idx, label in enumerate(labels):
        if label == i:
            clustered_points_tree.Add(points[idx], path)

# Create a list of cluster numbers for each index
point_clusters = [int(label) for label in labels]

# Output
clusters = clustered_points_tree
cluster_labels = point_clusters
