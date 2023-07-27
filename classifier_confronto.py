from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, matthews_corrcoef
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from time import time

emded_df = '../data/embedding_entire.pkl'
out_df = '../output/comp_entire.pkl' # classification three classes
out_df_nn = '../output/comp_nn_entire.pkl' #classification without neutral

def undersampling(df, random_state):
    # undersampling function, since label classes are unbalancd (ProVax is the smallest)
    l = df.annotation.value_counts()['ProVax']
    out = pd.concat([
        df[df.annotation == 'ProVax'],
        df[df.annotation == 'AntiVax'].sample(l, random_state=random_state),
        df[df.annotation == 'Neutral'].sample(l, random_state=random_state)
    ], ignore_index=True)
    return out

def classification(df_, classifier, iteration, name, neutral = True, return_clf = False):
    acc = []
    mcc = []
    acc_train = []
    mcc_train = []
    f1 = []
    for i in range(iteration):
        df = undersampling(df_, random_state = i)
        if not neutral:
            df = df[df.annotation != 'Neutral']

        tweet_emb = pd.DataFrame(df['tweet_embeddings'].to_list())
        X_train, X_test, Y_train, Y_test = train_test_split(tweet_emb,df['annotation'], 
                                                            shuffle=True,
                                                            test_size=0.3, random_state = 15)
        classifier.fit(X_train, Y_train)
        
        if return_clf:
            return classifier
        
        prediction = classifier.predict(X_test)
        
        f1.append(f1_score(Y_test,prediction,average=None))
        acc.append(accuracy_score(Y_test,prediction))
        mcc.append(matthews_corrcoef(Y_test,prediction))
        
        #faccio predizioni anche sul train set:
        prediction = classifier.predict(X_train)
        acc_train.append(accuracy_score(Y_train,prediction))
        mcc_train.append(matthews_corrcoef(Y_train,prediction))
        
        #faccio predizioni anche sul train set:
    out_dict = {'acc':acc,'mcc':mcc}
    for n,classes in enumerate(classifier.classes_):
        out_dict[f'f1_{classes}'] = [f[n] for f in f1]
    out_dict['acc_train'] = acc_train
    out_dict['mcc_train'] = mcc_train
    out_df = pd.DataFrame(out_dict)
    out_df['classifier'] = name
    return out_df



if __name__ == '__main__':
    iteration = 30
    #load df
    df = pd.read_pickle(emded_df)
    
    #classification
    t=time()
    print('classification three classes')
    print('tree')
    tree = DecisionTreeClassifier()
    tree_df = classification(df,tree, iteration, 'tree')
    print('forest')
    forest = RandomForestClassifier()
    forest_df = classification(df,forest, iteration, 'forest')
    print('logistic_regression')
    lr = LogisticRegression(max_iter=500)
    lr_df = classification(df, lr, iteration, 'log_reg')
    
    all_df = pd.concat([tree_df,forest_df,lr_df],ignore_index=True)
    all_df.to_pickle(out_df)
    
    print(f'time: {(time()-t)/60:.1f} min')
    
    #now without neutral
    t= time()
    print('\nClassification pro-anti')
    print('tree')
    tree = DecisionTreeClassifier()
    tree_df = classification(df,tree, iteration, 'tree', neutral=False)
    print('forest')
    forest = RandomForestClassifier()
    forest_df = classification(df,forest, iteration, 'forest', neutral=False)
    print('logistic regression')
    lr = LogisticRegression(max_iter=500)
    lr_df = classification(df, lr, iteration, 'log_reg', neutral=False)
    
    all_df = pd.concat([tree_df,forest_df,lr_df],ignore_index=True)
    all_df.to_pickle(out_df_nn)
    
    print(f'time: {(time()-t)/60:.1f} min')