import networkx as nx
import igraph as ig
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np   
import pickle
from tqdm import tqdm 
from fa2 import ForceAtlas2

def compute_positions(Gx,filename):
    forceatlas2 = ForceAtlas2(
                        # Behavior alternatives
                        outboundAttractionDistribution=False,  # Dissuade hubs
                        linLogMode=False,  # NOT IMPLEMENTED
                        adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
                        edgeWeightInfluence=1.0,

                        # Performance
                        jitterTolerance=1.0,  # Tolerance
                        barnesHutOptimize=True,
                        barnesHutTheta=1.2,
                        multiThreaded=False,  # NOT IMPLEMENTED

                        # Tuning
                        scalingRatio=3,
                        strongGravityMode=False,
                        gravity=5,

                        # Log
                        verbose=True)
    print('computing force atlas positions for graph visualization')
    positions = forceatlas2.forceatlas2_networkx_layout(Gx, pos=None, iterations=100)
    with open(filename, 'wb') as file:
        pickle.dump(positions,file)

def plot_community(cluster_obj, ncoms, networkx_graph, position_filename, filename):
    alphab=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q',
        'R','S','T','U','V','W','X','Y','Z','AA','AB','AC','AD','AE','AF','AG']
    comtodraw=alphab[:ncoms]
    
    #selecting clusters bigger than two-element cluster
    com_lists=[cl for cl in cluster_obj.subgraphs() if len(cl.vs)>2]
    com_lists.sort(key=lambda x : len(x.vs),reverse=True)
    print('Nodes in first {} coms'.format(ncoms),sum([len(cl.vs) for cl in com_lists[:ncoms]]))

    dict_ids_community = {}
    
    #assing each id a mebership letter
    for com,letter in zip(com_lists[:ncoms],comtodraw):
        for twt_id in com.vs.get_attribute_values('id'):
            dict_ids_community[twt_id] = letter
            
    #inserting other node only if degree greater than 2 (just for plotting)
    for com in com_lists[ncoms:]:
        for twt_id in com.vs.get_attribute_values('id'):
            if networkx_graph.degree(twt_id) >= 2: 
                dict_ids_community[twt_id] = alphab[ncoms]

    user_to_plot=list(dict_ids_community.keys())
    print(len(user_to_plot), 'users assigned to top {} communities and weight>=2'.format(ncoms))

    sizes=np.array([networkx_graph.degree(u) for u in user_to_plot])
    sizes=np.interp(sizes, (sizes.min(), sizes.max()), (10, 500))
    
    with open(position_filename,'rb') as file:
        positions = pickle.load(file)
    
    xfa=[positions[u][0] for u in user_to_plot]
    yfa=[positions[u][1] for u in user_to_plot]

    com_ids={c:list(set([u for u,com in dict_ids_community.items() if com==c])) for c in comtodraw}

    com2size={c:len(com_ids[c])/len(networkx_graph.nodes) for c in comtodraw}
    
    print('size of each community',com2size)
    print('total users printed', sum(com2size.values()))
    
    com2col={'A':'blue','B':'red','C':'orchid','D':'deepskyblue','E':'orange',
        'F':'grey','G':'deeppink','H':'black','I':'yellow','J':'brown','K':'cyan',
        'L':'lime','M':'green','N':'slateblue'}
    colors=[com2col[dict_ids_community[u]] for u in user_to_plot]
    
    #plotting
    f=plt.figure(dpi=500,figsize=(10,10))
    plt.scatter(xfa,yfa,c=colors,s=sizes,alpha=.7,
                marker='.',edgecolors='black',linewidths=.1)

    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for com,color in com2col.items() if com!='XX']

    scritte=[c+': {:.1f}%'.format(100*com2size[c]) for c in comtodraw]

    plt.legend(markers, scritte, numpoints=1,loc='lower left',#'upper right',
              fontsize=14)
    f.savefig(filename)