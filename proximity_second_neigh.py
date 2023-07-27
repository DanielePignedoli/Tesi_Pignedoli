import networkx as nx
import igraph as ig
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np   
import pandas as pd
import pickle
import operator

data_filename = '../data/embedding_df_train1000000.pkl'
graph_filename = '../graph/G_weighted.graphml'
cluster_filename = '../graph/community.cluster' #pickle
output_name = '../data/proximity_seconds.pkl'
n_coms=10 #number of community to consider
order = 2 #proximity up to second neighbors

#load data
print('loading data')
df = pd.read_pickle(data_filename)
graph = ig.Graph.Read_GraphML(graph_filename)
with open(cluster_filename,'rb') as file:
    clusters = pickle.load(file)



membership = clusters.membership #merbership assign to each node
id2index = {graph.vs[index]['id'] : index for index in graph.vs.indices} #trasform twitter id to graph index 

def closeness(user_id, community_index, graph, order):
    node_index = id2index[user_id] 
    neighbors = graph.neighborhood(node_index, order=order ,mindist=1) #first and second order
    neigh_in_community = [membership[node] for node in neighbors if membership[node] == community_index] 
    return len(neigh_in_community)/len(neighbors)

#ordering community indeces on community sizes
print('ordering communities')
values, counts = np.unique(membership, return_counts=True)
membership_ordered = dict(sorted(zip(values, counts), key=operator.itemgetter(1), reverse=True))

top_com_indeces = list(membership_ordered.keys())[:n_coms]

print('computing proximity values for each user in labeled dataset')
#computing proximity for each node
df['proximity'] = df['user.id'].apply(lambda x: [closeness(x, i, graph,order) for i in top_com_indeces])

#saving new embedding
df.to_pickle(output_name)