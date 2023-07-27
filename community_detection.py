import networkx as nx
import igraph as ig
import pandas as pd
from time import time
from plot_community import plot_community, compute_positions
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

plotting = False
n_coms = 6
distribution_filename = '../output/community_sizes.png'
community_filename = '../output/community.png'
cluster_filename = '../graph/community.cluster'
position_filename = '../data/position.forceatlas'

#loading graph
print('loading graph')
Gx = nx.read_graphml("../graph/G_weighted.graphml")
G = ig.Graph.Read_GraphML("../graph/G_weighted.graphml")


#community detection
print('community detection algorithm')
com_of_G= G.community_label_propagation(weights='weight')

#community size distribution (no two-size cluster)
com_lists=[cluster for cluster in com_of_G.subgraphs() if len(cluster.vs)>2]
com_lists.sort(key=lambda x : len(x.vs),reverse=True)
sizes = [len(c.vs) for c in com_lists]

print('plotting cluster distribution')
f=plt.figure(figsize=(8,4))
plt.bar(range(1,31), sizes[:30])
plt.ylabel('Cluster dimension')
plt.xlabel('Cluster number')
f.savefig(distribution_filename)

#print ratios up to 90% of users
thr = 0.90
cumulative_ratio = 0
for i,com in enumerate(com_lists):
    len_com = len(com.vs)
    ratio = len_com/len(G.vs)
    cumulative_ratio += ratio
    print(f'Nodes in coms {i}: {len_com} -  fraction : {ratio*100:.2f}% - cumulative : {cumulative_ratio*100:.2f}%')
    if cumulative_ratio > thr:
        break
        
#plotting
if plotting:
    #t = time()
    #print('computing position forceatlas')
    #compute_positions(Gx,position_filename)
    #print(f'finish in time: {(time()-t)/60:.1f} min')
    
    t = time()
    print('plotting community')
    plot_community( cluster_obj = com_of_G,
                    ncoms = n_coms,
                    networkx_graph = Gx,
                    position_filename = position_filename,
                    filename = community_filename,
                  )
    print(f'finish in time: {(time()-t)/60:.1f} min')
    
#saving cluster object
with open(cluster_filename,'wb') as file:
    pickle.dump(com_of_G,file)




