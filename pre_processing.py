import re    # for regular expressions 
import nltk  # for text manipulation 
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
from time import time
import pandas as pd

##cleaning
def cleaner(tweet):
    links = re.findall(r"http\S+", tweet)
    mentions = re.findall(r"@\w+", tweet)
    for link in links:
        tweet = tweet.replace(link, "")  
    for mention in mentions:
        tweet = tweet.replace(mention, "")
    return tweet


##tokenizer
my_tokenizer = RegexpTokenizer('\w+')
def tokenize(text):
    return my_tokenizer.tokenize(text)


##Stopword
italian_stopwords = set(nltk.corpus.stopwords.words('italian'))
italian_stopwords.remove('non') #parola importante che compare in diversi tweet, cambia il senso del tweet
italian_stopwords.remove('ne') 
italian_stopwords.remove('contro')
def is_stopwords(token):
    if token in italian_stopwords:
        return True
    return False

#stemmer
stemmer_snowball = SnowballStemmer('italian')
def stem(token):
    return stemmer_snowball.stem(token)

def pre_processing(text):
    clean_tweet = cleaner(text)
    tokens = tokenize(clean_tweet)
    tokens = [token.lower() for token in tokens] # lowering case, avoid duplicate
    tokens = [token for token in tokens if not is_stopwords(token)] #remove stopwords
    tokens = [stem(token) for token in tokens]
    return tokens

if __name__ == '__main__':
    
    #parmas:
    filename = "../data/processed_df.pickle"
    data_path= '../data/df_restored_2021_06_01.pickle'

    data = pd.read_pickle(data_path)

    #preprocessing
    t =time()
    print('Preprocessing text\n')
    data['processed_text'] = data['text'].map(pre_processing)
    print(f'time: {(t- time())/60:.1f} min')

    #saving df
    data.to_pickle(filename)