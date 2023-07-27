 from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, matthews_corrcoef
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import MDS,Isomap, TSNE
from umap import UMAP


import pandas as pd
from time import time

emded_df = '../data/both_embeddings.pkl'
out_df = '../output/class_both_embeddings.pkl'
out_df_nn = '../output/class_nn_both_embeddings.pkl'
embed_mode = 'both' #both, tweet, or node
dim_reduction = None
#dim_reduction = PCA(n_components = 15)
#dim_reduction = MDS(n_components = 10, random_state = 42, normalized_stress = 'auto')
#dim_reduction = TSNE(n_components = 2)
#dim_reduction = UMAP(n_components=2)
#dim_reduction = Isomap(n_components=2, n_neighbors=8)

def undersampling(df, random_state):
    # undersampling function, since label classes are unbalancd (ProVax is the smallest)
    l = df.annotation.value_counts()['ProVax']
    out = pd.concat([
        df[df.annotation == 'ProVax'],
        df[df.annotation == 'AntiVax'].sample(l, random_state=random_state),
        df[df.annotation == 'Neutral'].sample(l, random_state=random_state)
    ], ignore_index=True)
    return out

def classification(df_, classifier, iteration,embeddings = 'tweet' ,neutral = True, dim_reduction = None):
    acc = []
    mcc = []
    f1 = []
    if dim_reduction:
        print('Dim reduction')
    for i in range(iteration):
        print(i, )
        df = undersampling(df_, random_state = i)
        if not neutral:
            df = df[df.annotation != 'Neutral']
            
        #chosing embeddings
        if embeddings == 'both':
            tweet_emb = pd.DataFrame(df['tweet_embeddings'].to_list())
            node_emb = pd.DataFrame(df['proximity'].to_list())
            train = pd.concat([tweet_emb, node_emb], axis=1, ignore_index= True)
        elif embeddings == 'tweet':
            train = pd.DataFrame(df['tweet_embeddings'].to_list())
        elif embeddings == 'node':
            train = pd.DataFrame(df['proximity'].to_list())
        else:
            return ' Select embedding: "both","tweet" or "node" '
        
        #dimensionality_reduction
        if dim_reduction:
            red_values = dim_reduction.fit_transform(train)
            train = pd.DataFrame(red_values, index = df.id)
            
        X_train, X_test, Y_train, Y_test = train_test_split(train,df['annotation'].values, 
                                                            shuffle=True,
                                                            test_size=0.3, random_state =i+15)
        classifier.fit(X_train, Y_train)
        
        prediction = classifier.predict(X_test)
        
        f1.append(f1_score(Y_test,prediction,average=None))
        acc.append(accuracy_score(Y_test,prediction))
        mcc.append(matthews_corrcoef(Y_test,prediction))
    
    out_dict = {'acc':acc,'mcc':mcc}
    for n,classes in enumerate(classifier.classes_):
        out_dict[f'f1_{classes}'] = [f[n] for f in f1]
    out_df = pd.DataFrame(out_dict)
    return out_df



if __name__ == '__main__':
    iteration = 30
    #load df
    df = pd.read_pickle(emded_df)
    
    #classification
    t=time()
    print('logistic_regression')
    lr = LogisticRegression(max_iter=500)
    lr_df = classification(df, lr, iteration, embeddings= embed_mode , dim_reduction = dim_reduction)
    
    lr_df.to_pickle(out_df)
    
    print(f'time: {(time()-t)/60:.1f} min')
    
    #now without neutral
    t=time()
    print('classification with logistic_regression')
    lr = LogisticRegression(max_iter=500)
    lr_df = classification(df, lr, iteration, embeddings = embed_mode, neutral = False, dim_reduction = dim_reduction)
    
    lr_df.to_pickle(out_df_nn)
    
    print(f'time: {(time()-t)/60:.1f} min')
    