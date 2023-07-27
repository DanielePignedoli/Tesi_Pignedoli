import gensim 
from pre_processing import pre_processing
import pandas as pd
import numpy as np
from time import time

label_datapath = '../data/labeled_df.pkl'
file_name = '../data/embedding_entire.pkl'
model_path = '../model/entire_ns5.gensim'

def tweet2vector(tokens, model):
    size = model.vector_size
    vec = np.zeros(size)
    count = 0
    errors = 0
    for word in tokens:
        try:
            vec += model.wv[word]
            count += 1.
        except KeyError:  # handling the case where the token is not in vocabulary
            errors += 1

    if count != 0:
        vec /= count

    return vec,errors

t = time()
print('loading model')
#load model
model_w2v = gensim.models.Word2Vec.load(model_path)
print('loading data')
#load label data
label_df = pd.read_pickle(label_datapath)
#preprocess text
label_df['processed_text'] = label_df['text'].map(pre_processing)
print('creating tweet embeddings')
#create embedding
label_df[['tweet_embeddings','embedding_errors']] = label_df['processed_text'].apply(lambda x : pd.Series(tweet2vector(x,model_w2v)))
#save df
label_df.to_pickle(file_name)
print(f'finish in time: {(time()-t)/60:.1f} min')

    