#! python3
#r: openpyxl, psycopg2, Ipython

import networkx as nx
import pandas as pd
import rhinoscriptsyntax as rs
import numpy as np

##### INPUTS ############################################################
nf_columns = n_features.columns
ef_columns = e_features.columns
nc_columns = n_classes.columns
ec_columns = e_classes.columns

################# ADD GEOMETRY (LINE SEGMENTS) AND CONVERT THEM TO DATAFRAMES IN A NETWORKX GRAPH ####################################
# Input: List of lines and original points (GUIDs)
network = [line for line in lines]  # Replace 'network' with your input parameter
original_point_guids = points

# Convert GUIDs to Point3d objects
original_points = [rs.coerce3dpoint(guid) for guid in original_point_guids]

# Create a mapping from points to node identifiers
point_to_node = { (pt.X, pt.Y, pt.Z): pt for pt in original_points }

# Create an empty graph
G = nx.Graph()

# Add edges to the graph
for line in network:
    start_point = rs.CurveStartPoint(line)
    end_point = rs.CurveEndPoint(line)

    start_point_tuple = (start_point.X, start_point.Y, start_point.Z)
    end_point_tuple = (end_point.X, end_point.Y, end_point.Z)
    
    if not G.has_node(start_point_tuple):
        G.add_node(start_point_tuple, pos=start_point_tuple)
    if not G.has_node(end_point_tuple):
        G.add_node(end_point_tuple, pos=end_point_tuple)
    
    G.add_edge(start_point_tuple, end_point_tuple)

# Create a DataFrame with nodes in the same order as original_points
node_positions = [(pt.X, pt.Y, pt.Z) for pt in original_points]
nodes_data = {pos: G.nodes[pos] for pos in node_positions if G.has_node(pos)}

# Convert nodes to DataFrame
nodes_df = pd.DataFrame(
    [(pos, nodes_data.get(pos, {})) for pos in node_positions],
    columns=['Nodes', 'Attributes']
)

# Extract x, y, z coordinates and add as separate columns
nodes_df['x'] = nodes_df['Attributes'].apply(lambda attr: attr['pos'][0])
nodes_df['y'] = nodes_df['Attributes'].apply(lambda attr: attr['pos'][1])
nodes_df['z'] = nodes_df['Attributes'].apply(lambda attr: attr['pos'][2])

# Save original order of nodes
nodes_df['df_order'] = nodes_df.index 

##Remove the original 'Attributes' column if no longer needed
nodes_df.drop(columns=['Attributes'], inplace=True)

# Convert edges to DataFrame
edges_df = pd.DataFrame(list(G.edges(data=True)), columns=['Source', 'Target', 'Attributes'])

# Add additional columns from other DataFrames (if applicable)
if 'nf_columns' in locals() and 'n_features' in locals():
    nodes_df[nf_columns] = n_features[nf_columns]
if 'ef_columns' in locals() and 'e_features' in locals():
    edges_df[ef_columns] = e_features[ef_columns]
if 'nc_columns' in locals() and 'n_classes' in locals():
    nodes_df["classes"] = n_classes[nc_columns]
if 'ec_columns' in locals() and 'e_classes' in locals():
    edges_df["classes"] = e_classes[ec_columns]

###################### ASSIGN ATTRIBUTES FOR GRAPH ###############################################################
# Inspect DataFrame columns and types
def inspect_dataframe(df):
    print("DataFrame inspection:")
    print(df.head())
    print(df.dtypes)
    for column in df.columns:
        if df[column].apply(lambda x: isinstance(x, (list, dict, np.ndarray))).any():
            print(f"Column '{column}' contains sequences or non-numeric data.")
    print("\n")

inspect_dataframe(nodes_df)
inspect_dataframe(edges_df)

# Convert DataFrame columns to float32
def convert_columns_to_float32(df, exclude_columns):
    for column in df.columns:
        if column not in exclude_columns:
            try:
                # Ensure column data is numeric
                df[column] = pd.to_numeric(df[column], errors='coerce')
                # Convert to float32
                df[column] = df[column].astype(np.float32)
            except Exception as e:
                print(f"Error converting column '{column}': {e}")
    return df

# Convert node and edge DataFrames to float32
nodes_df = convert_columns_to_float32(nodes_df, exclude_columns=['Nodes'])
edges_df = convert_columns_to_float32(edges_df, exclude_columns=['Source', 'Target'])

# Convert class columns to long integers
def convert_classes_to_long(df, class_columns):
    for column in class_columns:
        if column in df.columns:
            try:
                df[column] = df[column].astype(np.int64)
            except Exception as e:
                print(f"Error converting column '{column}': {e}")
    return df

# Convert classes to long integers
class_columns = ['classes']  # Add other class columns if needed
nodes_df = convert_classes_to_long(nodes_df, class_columns)
edges_df = convert_classes_to_long(edges_df, class_columns)

# Assign node attributes
for index, row in nodes_df.iterrows():
    node = row['Nodes']
    attributes = row.drop('Nodes').to_dict()
    G.nodes[node].update(attributes)

# Assign edge attributes
for index, row in edges_df.iterrows():
    source = row['Source']
    target = row['Target']
    attributes = row.drop(['Source', 'Target']).to_dict()
    G.edges[source, target].update(attributes)

# Print node and edge attributes to verify
print("Node attributes:")
for node in G.nodes:
    print(node, G.nodes[node])

print("\nEdge attributes:")
for edge in G.edges:
    print(edge, G.edges[edge])

##Remove the original 'Nodes' column if no longer needed
nodes_df.drop(columns=['Nodes'], inplace=True)

####################### OUTPUTS ######################################################################
# Output DataFrames
G_nodes = nodes_df
G_edges = edges_df
# Graph = G  # Uncomment if you need to output the graph itself

print(nodes_df.columns)
