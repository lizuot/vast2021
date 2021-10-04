import warnings
warnings.filterwarnings('ignore')
import glob
import csv
import nltk
nltk.download('stopwords')
import sklearn
import re
import numpy as np
from sklearn.cluster import DBSCAN
from collections import Counter
import pandas as pd
import spacy
from pprint import pprint

import pandas
import bokeh

import glob
import os

import umap
#import umap.plot

from yellowbrick.text import TSNEVisualizer
from nltk.corpus import stopwords as sw
stop_words = sw.words('english')
stop_words.extend(['totally','comes','ur','xx','ya','thru','wasnt','either','gg','went','fyi','thing','theyd','tha','ha','weve','yet','yup','totaly','gonna','em','anyway','finally','aw','lets','exactly','rt','additional','area','become','cant','day','department','didnt','dont','dude','everything','fall','find','getting','give','going','got','heading','hi','im','let','like','likely','made','might','near','never','oh','ok','one','people','person','place','please','put','recommend','says','seems','settle','site','still','tag','theres','things','us','usual','wont','youre',"you've",'from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])

com1 = ['boom', 'exlosion_heard', 'explosions']
com2 = ['pok', 'rally']
com3 = ['abilapost', 'abilafire']
com4 = ['abd', 'officer','officers','cops','cop']
com5 = ['reported']
com6 = ['AbilasFinest','abdheroes']
com7 = ['dancing','dolphin']
com8 = ['paramedics','ambulance']
com9 = ['park']
com10 = ['apartment_complex']  

from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")

from bokeh.io import show, output_file
from bokeh.plotting import figure
from bokeh.layouts import gridplot


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import pickle
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.test.utils import common_corpus, common_dictionary
from gensim.models import HdpModel
from gensim.models import CoherenceModel

# Plotting tools
import pyLDAvis
#import pyLDAvis.gensim  # don't skip this
import pyLDAvis.sklearn


def stem(word):
    return stemmer.stem(word).strip()

def prep(corp):  
    # tokenization
    tokenized = tokenize(corp)
    # stopwords
    cleaned = [word for word in tokenized if word not in stopwords and word is not '']
    # steming
    stemed = [stem(word) for word in cleaned]
    return stemed

def tokenize(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            clean_word = regex.sub('', word)
            tokens.append(clean_word.lower())
    return tokens

def h(stem):
    data = []
    for w in set(stem):
        data.append((stem.count(w), w))
    data.sort(reverse=True)
    return data

def xy(data):   
    #l = range(len(data))
    l = range(20)
    x = [data[i][1] for i in l]
    y = [data[i][0] for i in l]
    return x,y


def plot(x,y, title):
    p = figure(x_range=x, plot_width=1200, plot_height=400, title=title,
               toolbar_location=None, tools="")

    p.vbar(x=x, top=y, width=0.9, line_color="white", fill_color="navy", alpha=0.5)

    p.xgrid.grid_line_color = None
    p.xaxis.major_label_orientation = 1.2
    p.y_range.start = 0
    return p



def sent_to_words(sentences):
    for sent in sentences:
        sent = re.sub('\S*@\S*\s?', '', sent)  # remove emails
        sent = re.sub('\s+', ' ', sent)  # remove newline chars
        sent = re.sub("\'", "", sent)  # remove single quotes
        sent = gensim.utils.simple_preprocess(str(sent), deacc=True) 
        yield(sent)  

        # !python3 -m spacy download en  # run in terminal once
def process_words(texts, stop_words=stop_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """Remove Stopwords, Form Bigrams, Trigrams and Lemmatization"""
    texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    texts = [bigram_mod[doc] for doc in texts]
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
    terms_only = texts
    #print(terms_only)
    
    for j in range(len(terms_only)):
        for i in range(len(terms_only[j])):
            if terms_only[j][i] in com1:
                terms_only[j][i] = 'explosion'
            if terms_only[j][i] in com2:
                terms_only[j][i] = 'pokrally'
            if terms_only[j][i] in com3:
                terms_only[j][i] = 'abila'
            if terms_only[j][i] in com4:
                terms_only[j][i] = 'police'
            if terms_only[j][i] in com5:
                terms_only[j][i] = 'reports'
            if terms_only[j][i] in com6:
                terms_only[j][i] = 'afd'
            if terms_only[j][i] in com7:
                terms_only[j][i] = 'dancing_dolphin'
            if terms_only[j][i] in com8:
                terms_only[j][i] = 'ambulance'
            if terms_only[j][i] in com9:
                terms_only[j][i] = 'city_park'
            if terms_only[j][i] in com10:
                terms_only[j][i] = 'apartment'
            
            
    texts_out = []
    #nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    #for sent in texts:
        #doc = nlp(" ".join(sent)) 
        #texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    # remove stopwords once more after lemmatization
    #texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]    
    #return texts_out
    return terms_only


labels = []
corpus1 = []
corpus = ""
corpus2 = []

df = pd.read_csv('./data/all.csv')
data = df.message.values.tolist()
regex = re.compile('[^a-zA-Z]')
count_all = Counter()
data_words = list(sent_to_words(data))
# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)
        
corpus2 = process_words(data_words)  # processed Text Data!
corpus3 = []
for li in corpus2:
    string = ' '.join(li)
    corpus3.append(string)
    
print(corpus3)
from sklearn.feature_extraction.text import TfidfVectorizer  
tfidfconverter = TfidfVectorizer(max_features=2000, min_df=5, max_df=0.7, stop_words=stop_words)  
X = tfidfconverter.fit_transform(corpus3).toarray()

tfidf_embedding = umap.UMAP(metric='hellinger').fit(X)
#fig = umap.plot.points(tfidf_embedding, labels=hover_df['category'])
fig = umap.plot.points(tfidf_embedding)

tsne = TSNEVisualizer()
tsne.fit(X, y)
tsne.show()