import networkx as nx
import pandas as pd
from time import time

t=time()

print('load data e preprocess dataframe for network creation')
#load data
data = pd.read_pickle('../data/df_restored_2021_06_01.pickle')
#select uesrs columns
df_users = data[['id','user.id', 'retweeted_status.user.id']].copy()
df_users.dropna(how='any', inplace=True)

#select link bigger than weight 1
counts = df_users.groupby(['user.id', 'retweeted_status.user.id']).size()
valid_counts = counts[counts >= 2].reset_index()
valid_counts.rename(columns={0:'weight'}, inplace=True)

print('creating network, undirected and weighted')

G=nx.from_pandas_edgelist(valid_counts, source = 'retweeted_status.user.id', target = 'user.id', edge_attr='weight', create_using=nx.Graph())

largest_cc = max(nx.connected_components(G), key=len)
G_connected = G.subgraph(largest_cc)

print('selecting largest connected components')

print('saving graphml')
nx.write_graphml(G_connected,'../graph/G_weighted.graphml',prettyprint=False)

Gx = nx.read_graphml("../graph/G_weighted.graphml")
print(Gx)
print(f'finish in time: {(time()-t)/60:.1f} min')





