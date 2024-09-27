#! python3
#r: openpyxl, psycopg2, Ipython

import dgl
import torch
import networkx as nx

# Assuming G is your NetworkX graph
# Ensure the graph is undirected before conversion
if G.is_directed():
    raise ValueError("NetworkX graph is directed. Please provide an undirected graph.")

print("Is the networkX graph directed?", G.is_directed())


# Manually add reverse edges for DGL
edges = list(G.edges())
reverse_edges = [(v, u) for u, v in edges]
G.add_edges_from(reverse_edges)

# Convert to DGL graph
nodes_attributes_list = G_nodes.columns.tolist()
graph = dgl.from_networkx(G, node_attrs=nodes_attributes_list)


# Additional processing for node attributes
columns_to_exclude = ['x', 'y', 'z', 'df_order', 'classes']
for i in nodes_attributes_list:
    graph.ndata[i] = torch.reshape(graph.ndata[i], (graph.ndata[i].shape[0], 1))

selected_attributes = [attr for attr in nodes_attributes_list if attr not in columns_to_exclude]
if len(selected_attributes) > 0:
    graph.ndata["feat"] = torch.cat([graph.ndata[i] for i in selected_attributes], 1)
    graph.ndata["feat"] = graph.ndata["feat"]
else:
    raise ValueError("No attributes selected for concatenation. Check your nodes_attributes_list slice.")

# Verify if the graph is now undirected
edges = graph.edges()
edge_set = set(zip(edges[0], edges[1]))
is_directed = len(edge_set) != graph.num_edges()

print("Is the DGL graph directed after conversion?", is_directed)


################## OUTPUTS
dgl_graph = graph
graph_nodes = (graph.nodes())
graph_edges = (graph.edges())
