# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 19:55:25 2021

@author: Andrei
"""
import numpy as np 
import pandas as pd 

import re
import string
import nltk
from nltk.corpus import stopwords

from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
disaster_tweets = train[train['target']==1]['text']
non_disaster_tweets = train[train['target']==0]['text']

# Text cleaning 
def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def text_preprocessing(text):
    """
    Cleaning and parsing the text.

    """
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    
    nopunc = clean_text(text)
    tokenized_text = tokenizer.tokenize(nopunc)
    remove_stopwords = [w for w in tokenized_text if w not in stopwords.words('english')]
    combined_text = ' '.join(remove_stopwords)
    return combined_text

# =============================================================================
# train['text_prep'] = train['text'].apply(lambda x: text_preprocessing(x))
# test['text_prep'] = test['text'].apply(lambda x: text_preprocessing(x))
# =============================================================================

train['text_prep'] = train['text'].map(text_preprocessing)
test['text_prep'] = test['text'].map(text_preprocessing)

# Lemmatizer
def lemmatizer(text):
    tokenizer = nltk.tokenize.TreebankWordTokenizer()
    tokens = tokenizer.tokenize(text)
    lemmatizer=nltk.stem.WordNetLemmatizer()
    lemm_text = [lemmatizer.lemmatize(w) for w in tokens]
    combined_text = ' '.join(lemm_text)
    return combined_text

train['lemm_text'] = train['text_prep'].map(lemmatizer)
test['lemm_text'] = test['text_prep'].map(lemmatizer)

# Bag of Words 
count_vectorizer = CountVectorizer()
train_vectors = count_vectorizer.fit_transform(train['lemm_text'])
test_vectors = count_vectorizer.transform(test["lemm_text"])

# =============================================================================
# tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))
# train_tfidf = tfidf.fit_transform(train['text_prep'])
# test_tfidf = tfidf.transform(test["text_prep"])
# =============================================================================

# Fitting a simple Naive Bayes
model = MultinomialNB()
model.fit(train_vectors, train["target"])

scores = model_selection.cross_val_score(model, train_vectors, train["target"], cv=5, scoring="f1")
scores
print(f"Model cross validation mean F1 score:{scores.mean(): .2f}")

# Submission
submission = pd.read_csv('sample_submission.csv')
submission["target"] = model.predict(test_vectors)
submission.to_csv("submission.csv", index=False)
