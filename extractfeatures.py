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
from nltk.corpus import genesis
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from nltk.corpus import wordnet as wn
from sklearn import decomposition
import string

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

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
    stop_words.add("I")

    stop_words.remove("not") # not may be important for analysis
    reviews_no_stop = []
    for review in reviews:
        
        ## recover sentence
        tokens = review.split()
        review_no_stop_string = ""
        for token in tokens:
            token = token.lower()
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


def tag_hypernyms_with_adjustments(reviews):
    pass
    """ 
    Goal: drop hypernyms that don't matter
    """

def food_serv_sim_vec(reviews, test = False, vtest = False):
    reviews = drop_stop_words(reviews) # drop the stop words as a preprocessing step
    food_words = wn.synsets("food") # as in "food or drink"
    service_words = [wn.synsets("waiter")[0],wn.synsets("service")[14]] # this version of service is what we meant
    location_word = wn.synsets("location")[0] # to avoid things like diner, restaurant, etc. 
    result_word = wn.synsets("result")[0] # gets rid of "worst"
    final_list = [] # this will be returned
    for review in reviews:
        review = review.split() # tokenize each review
        noun_count = 0
        food_scores = []
        service_scores = []
        for word in review:
            #preprocess the token
            punct_strip = str.maketrans('', '', string.punctuation)
            word = word.translate(punct_strip)
            word = word.lower()
            
            # Crete arrays to fill with similarity values
            # we do this because we will have lots of  words in the synset, and we will grab the 
            #max similarity value for each of these
            similarities_food = []
            similarities_serv = []
            similarities_loc = []
            similarities_result= []

            
            syns = wn.synsets(word, pos="n") # we only want to look at nouns because 
            # a) they are most informative and b) wup trips up on other parts of speech
            
            
            if len(syns) > 0:
                noun_count = noun_count + 1

                for w1 in syns:
                    similarities_food.append(max( w1.wup_similarity(food_words[0]),
                                                 w1.wup_similarity(food_words[1]))) 
                    similarities_serv.append(max(w1.wup_similarity(service_words[0]), 
                                                 w1.wup_similarity(service_words[1])))
                    similarities_loc.append( w1.wup_similarity(location_word))
                    similarities_result.append(w1.wup_similarity(result_word))

                     #word_vec.append([])
                sim_food = max(similarities_food)
                sim_serv = max(similarities_serv)
                sim_loc = max(similarities_loc)
                sim_res = max(similarities_result)
                
                if(vtest):
                    print(word + " FOOD: " + str(sim_food) + " SERV: " + str(sim_serv) + " LOC " + str(sim_loc))
                    print("----")
                if sim_loc > sim_serv and sim_loc > sim_food:
                    pass
                elif sim_res > sim_serv and sim_res > sim_food:
                    pass
                elif sim_food > sim_serv + .3 or sim_food > .6: # experimentally, food items have this high similarity, but sometimes come to close to service, so we adjust
                    food_scores.append(10*(sim_food))
                    if(test): 
                        print("food: " + word) 
                elif sim_serv > sim_food + .3:
                    service_scores.append(10*(sim_serv))
                    if (test):
                        print("service: " + word)

        if noun_count: 
            final_list.append([sum(food_scores)/noun_count, sum(service_scores)/noun_count])
        else:
            final_list.append([sum(food_scores), sum(service_scores)])

    return final_list

    
def food_serv_sim_vec_nationality(reviews):
    pass    

def reward_food_words(reviews):
    foods = []
    food_adjectives = []
    pass

def tag_embedding(reviews):
    pass
    
def working_space(reviews): ## JUST RANDOM PLACE TO MESS AROUND
    big_array = bag_words(reviews)
    w_norm = np.normalize(big_array)
    
    
    h = food_serv_sim_vec(reviews)
    dt=np.dtype('float','float') 
    h_1 = np.array(h,dtype=dt)
    norm_h = normalize(h_1)
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(h_1)
    y_km = kmeans.fit_predict(norm_h)

        
    plt.scatter(x=h_1[:,0], y=h_1[:,1], c=y_km)    
    
    big_array = big_array.astype(float)
    final_big_array = np.array(float)
    
    
    for i in range(len(big_array)):
         x = np.append(big_array[i], [h_1[i,0], h_1[i,1]])
         if i == 0:
            final_big_array = x
         else:
            final_big_array = np.vstack([final_big_array, x])
    
    f_norm = normalize(final_big_array)
    kmeans = KMeans(n_clusters=2)
    pca2D = decomposition.PCA(2)
    plot_columns = pca2D.fit_transform(f_norm)
    
    
    kmeans.fit(f_norm)
    y_km = kmeans.fit_predict(f_norm)
    plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=y_km)    

    for i in range(370,500):
        x = food_serv_sim_vec([reviews[i]])
        if (x[0][0] > x[0][1]):
            print(reviews[i])
            print(x)
            break
    
if __name__ == "__main__":
    reviews = load_review_text(1, 500)
    