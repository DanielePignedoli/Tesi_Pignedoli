import gensim  #embedding library
from callback_loss import LossLogger
from time import time
import pandas as pd

#parmas:
epochs = 20
SIZE = 200
NEGATIVES = 5
MIN_COUNT = 3
train_lenght = 5000000 #number of documents used to train model
filename = "../model/entire_ns5.gensim"
data_path= '../data/processed_df.pickle'


data = pd.read_pickle(data_path)
processed_text = data.sample(train_lenght,random_state=3)['processed_text']

#loss logger
loss_logger = LossLogger()

print('Parameters')
print(f'epochs: {epochs}\nVector_size: {SIZE}\nNeg_samples: {NEGATIVES}\nMin_count: {MIN_COUNT}\n Train_Data: {train_lenght}\n\n')
print('Building model')
### building model 
model_w2v = gensim.models.Word2Vec(
        processed_text,
        size=SIZE, # desired no. of features/independent variables
        window=5, # context window size
        min_count=MIN_COUNT, # Ignores all words with total frequency lower than 2.                                  
        sg = 1, # 1 for skip-gram model
        hs = 0,
        negative =NEGATIVES, # for negative sampling
        workers= 32, # no.of cores
        seed = 34,
        )
    
### trainig model
print('Training model')
t = time()
model_w2v.train(processed_text, total_examples= train_lenght, epochs=epochs,callbacks=[loss_logger], compute_loss=True)
print(f'time: {(time()-t)/60:.1f} min')

### saving model
with open(filename,'wb') as file:
    model_w2v.save(file)
    


    

        
        