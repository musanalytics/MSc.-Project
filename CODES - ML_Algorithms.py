#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: hunchoahmad
"""

#importing necessary libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import re 
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk import word_tokenize
from nltk.corpus import stopwords



# model building
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from tensorflow.keras.preprocessing.text import Tokenizer

# metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import warnings
%matplotlib inline
warnings.filterwarnings('ignore')

#read and understand the data
train = pd.read_csv("/Users/hunchoahmad/Downloads/Corona_NLP_train.csv", encoding="latin-1") 
test = pd.read_csv("/Users/hunchoahmad/Downloads/Corona_NLP_test.csv", encoding="latin-1")
train.head()

test.head()

print(train.info(),'\n')
test.info()

train.drop_duplicates(inplace=True)
test.drop_duplicates(inplace=True)

train.drop(['UserName', 'ScreenName'], axis=1, inplace=True)
test.drop(['UserName', 'ScreenName'], axis=1, inplace=True)

# combine train and test dataframes
combined = pd.concat([train, test], ignore_index= True)

# select relevant features: tweet and Sentiments
combined = combined.loc[:, ["OriginalTweet", "Sentiment"]]

# load stop words
stop_word = stopwords.words('english')

def clean_tweet(text):
    text = re.sub(r"#\w+", " ", text)            # remove hashtags
    text = re.sub(r"@\w+", " ",text)             # remove mentions
    text = re.sub(r"http\S+", " ", text)         # remove urls
    text = re.sub(r"[^a-zA-Z]", " ", text)        # remove non-words (digits, punctuations etc)
    text = text.lower().strip()                  # convert tweet to lowercase and strip
    
    text = " ".join([word for word in text.split() if not word in stop_word])           # remove stop words    
    
    text = " ".join(nltk.word_tokenize(text))           # tokenize text
      
    return text

# clean OriginalTweet and assign the data to an new "tweet" column
combined['tweet'] = combined['OriginalTweet'].apply(lambda x: clean_tweet(x))

# encode Sentiment label values
le = LabelEncoder()
combined.Sentiment = le.fit_transform(combined.Sentiment)

# split data back into training and validation sets and sets
train = combined[: len(train)]
test = combined[len(train):].reset_index(drop=True)

# split test test set
X_test = test.tweet
y_test = test.Sentiment


# split training set into training and validation set
X_train, X_val, y_train, y_val = train_test_split(train.tweet,train.Sentiment, test_size=0.2,random_state=42)


# initialize vectorizer
vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,2), min_df=5).fit(X_train)

X_train = vectorizer.transform(X_train)
X_val = vectorizer.transform(X_val)
X_test = vectorizer.transform(X_test)

# intializing the model and fitting it on the training data
logmodel = LogisticRegression(max_iter=10000)
logmodel.fit(X_train, y_train)

# checking training accuracy achieved
cross_val_score(logmodel, X_train, y_train, cv=5, verbose=1, n_jobs=-1).mean()

# extract labels from encoder
labels = list(le.classes_)
print(classification_report(val_pred, y_val, target_names= labels), '\n')

# make predictions
val_pred = logmodel.predict(X_val)
test_pred = logmodel.predict(X_test)

# print classification report
print(classification_report(test_pred, y_test, target_names= labels))

# check test accuracy
print('accuracy score on validation set: ', accuracy_score(y_val, val_pred))
print('accuracy score on test set:', accuracy_score(y_test, test_pred))

#implemnting solution using naive bayaes algorithm
from sklearn import naive_bayes
model=naive_bayes.MultinomialNB()
model.fit(X_train, y_train)

print(accuracy_score(y_test,predNB))

print(classification_report(y_test,predNB,target_names= labels))

from sklearn.ensemble import RandomForestClassifier
RFC=RandomForestClassifier(random_state=0)
RFC.fit(X_train,y_train)

predrfc=RFC.predict(X_test)
print(accuracy_score(y_test,predrfc))

print(classification_report(y_test,predrfc,target_names= labels))



