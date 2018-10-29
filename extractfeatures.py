#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 17:00:32 2018

@author: fraifeld-mba
"""
import pandas as pd 
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet as wn
import string

def parse_json(line):
    j = json.loads(line)
    return j

def load_data(i=1, k=150000): ## loads data indexed from i to k.  Defaults to whole set
    df = pd.DataFrame()
    with open("merge2.json","r") as f:
        count = 1
        for line in f:
            if count < i:
                pass
            elif count < k+1:
                df_plus = pd.read_json(line, lines = True)
                df = df.append(df_plus)
                
            else:
                break
            count = count + 1
    return df.reset_index()

def drop_stop_words(reviews): # drops stop words
    stop_words = set(stopwords.words('english')) 
    stop_words.add("The")
    stop_words.add("I")

    stop_words.remove("not") # not may be important for analysis
    reviews_no_stop = []
    for review in reviews:
        
        ## recover sentence
        tokens = review.split()
        review_no_stop_string = ""
        for token in tokens:
            if token not in stop_words:
                if len(review_no_stop_string) == 0:
                    review_no_stop_string = token
                elif len(token) > 1 :
                    review_no_stop_string = review_no_stop_string + " " + token
                else:
                    review_no_stop_string = review_no_stop_string + token
        reviews_no_stop.append(review_no_stop_string)
    return reviews_no_stop

def stem(reviews):
    ps = PorterStemmer()
    reviews_stemmed = []

    for review in reviews:
        tokens = review.split()
        stemmed_sentence = ""
        for token in tokens:
            token = ps.stem(token)
            if len(stemmed_sentence) == 0:
                stemmed_sentence = token
            elif len(token) > 1:
                stemmed_sentence = stemmed_sentence + " " + token
            else:
                stemmed_sentence = stemmed_sentence + token
        reviews_stemmed.append(stemmed_sentence)
    return reviews_stemmed

def load_review_text(i, k):
    df = load_data(i, k)
    return df["text"]

def bag_words(reviews):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(reviews)
    X = X.toarray()   
    return X

def preprocess_hypernyms(h):
       l = [' '.join(x) for x in h]
       #df = pd.DataFrame(columns = ['hypernyms'])
       #df["hypernyms"] = l
       return l

def get_unigram_frequencies(reviews):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(reviews)
    return (list(zip(vectorizer.get_feature_names(), np.ravel(X.sum(axis=0)))))


def print_ith_freq_unigram(reviews, i=1):
    x = get_unigram_frequencies(reviews)
    print(sorted(x,key=lambda item:item[1])[-i])
    

def bigrams(reviews):
    vectorizer = CountVectorizer(ngram_range = (1,2), token_pattern = r'\b\w+\b', min_df=1)
    X = vectorizer.fit_transform(reviews)
    X = X.toarray()   
    return X

def get_bigram_frequencies(reviews):
     vectorizer = CountVectorizer(ngram_range = (1,2), token_pattern = r'\b\w+\b', min_df=1)
     X = vectorizer.fit_transform(reviews)
     return (list(zip(vectorizer.get_feature_names(), np.ravel(X.sum(axis=0)))))

def weight_by_tfidf(reviews):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(reviews)
    X = X.toarray()   
    return X                    

def get_tfids(reviews):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(reviews)
    return (list(zip(vectorizer.get_feature_names(), np.ravel(X.mean(axis=0)))))
    
def print_ith_highest_tfidf(reviews, i=1):
    x = get_tfids(reviews)
    print(sorted(x,key=lambda item:item[1])[-i])
    

 
def tag_hypernyms(reviews):
    reviews = drop_stop_words(reviews)
    full_list_of_hypernums = []
    for review in reviews:
        tokens = review.split()
        review_hypernyms = []
        for token in tokens:
            punct_strip = str.maketrans('', '', string.punctuation)
            token = token.translate(punct_strip)
            token_syns = wn.synsets(token.lower())
            hypernyms = []
            if len(token_syns) > 0:
                hypernyms = token_syns[0].hypernyms() # This needs to be justifed
            if len(hypernyms) > 0:
                if "_" in hypernyms[0].lemma_names()[0]:
                    hnym = hypernyms[0].lemma_names()[0].split("_")
                    review_hypernyms.append(hnym[len(hnym)-1])

                else:
                    review_hypernyms.append(hypernyms[0].lemma_names()[0])
        full_list_of_hypernums.append(review_hypernyms)
    return full_list_of_hypernums


## OPTIONAL
def tag_embedding(reviews):
    pass
    
if __name__ == "__main__":
    reviews = load_review_text(1, 800)
    l = tag_hypernyms(reviews)
    l_proc = preprocess_hypernyms(l)
    