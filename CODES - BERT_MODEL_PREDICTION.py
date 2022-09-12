#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: hunchoahmad
"""

pip install --upgrade tensorflow_hub
pip install tensorflow-text
pip install -q tf-models-official
import os
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

from official.nlp import optimization
from nltk.corpus import stopwords

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow.keras import layers, losses, preprocessing

TRAIN_PATH = "/Users/hunchoahmad/Downloads/Corona_NLP_train.csv"
TEST_PATH = "/Users/hunchoahmad/Downloads/Corona_NLP_test.csv"
CLASSES = [ 'Negative', 'Positive', 'Neutral', 'Extremely Positive', 'Extrmely Negative']
BATCH_SIZE = 128
EPOCHS = 16
LEARNING_RATE = 1e-05 #small gradient steps to prevent forgetting in transfer learning.

# MODEL_NAME = 'bert-base-uncased'
tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1'
tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'

train_data = pd.read_csv(TRAIN_PATH, encoding='L1')
test_data = pd.read_csv(TEST_PATH,encoding='L1')

train_data.head()

# replace "extremely positive/negative" with "postive/negative"
train_data["Sentiment"] = train_data["Sentiment"].str.replace("Extremely Negative", "Negative")
train_data["Sentiment"] = train_data["Sentiment"].str.replace("Extremely Positive", "Positive")

test_data['Sentiment'] = test_data.Sentiment.str.replace('Extremely Positive', 'Positive')
test_data['Sentiment'] = test_data.Sentiment.str.replace('Extremely Negative', 'Negative')

# -- Pre-processing -- 

# drop empty tweets, and unclassified tweets.
train_data.OriginalTweet.dropna()
train_data.Sentiment.dropna()
test_data.OriginalTweet.dropna()
test_data.Sentiment.dropna()

# remove stop-words
sw_nltk = stopwords.words('english')
func = lambda text : " ".join([word for word in str(text).split() if word.lower() not in sw_nltk])
train_data['OriginalTweet'] = train_data['OriginalTweet'].apply(func)

# TODO: remove Urls and HTML links

# -- Split Data to train, validation and test -- 
train_X, val_X, train_y, val_y = model_selection.train_test_split(train_data['OriginalTweet'],
                                                                  train_data['Sentiment'], 
                                                                  test_size=0.2)

test_X, test_y = test_data['OriginalTweet'],test_data['Sentiment']
# -- convert labels to one hot --
label_encoder = LabelEncoder()

vec = label_encoder.fit_transform(train_y)
train_y = tf.keras.utils.to_categorical(vec)

vec = label_encoder.fit_transform(val_y)
val_y = tf.keras.utils.to_categorical(vec)

vec = label_encoder.fit_transform(test_y)
test_y = tf.keras.utils.to_categorical(vec)

def bert_text_classification():

    # - text input -
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        
    # - preprocessing layer - 
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
        
    # - encoding - 
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
        
    # - output -
    outputs = encoder(encoder_inputs)
        
    # - classifier layer -
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.2)(net)
    net = tf.keras.layers.Dense(3, activation='softmax', name='classifier')(net)
    
    model = tf.keras.Model(text_input, net)
    return model
        
model = bert_text_classification()   

pip install graphviz

test_text = ['some random tweet']
bert_raw_result = model(tf.constant(test_text))

# -- Model structure -- 
tf.keras.utils.plot_model(model)

# -- Loss -- 
loss = tf.keras.losses.CategoricalCrossentropy()

# Optimizing 
# "Adaptive Moments" (Adam) optimizer. 
train_data_size = len(train_X)
steps_per_epoch = int(train_data_size/BATCH_SIZE)
num_train_steps = steps_per_epoch * EPOCHS
num_warmup_steps = int(0.1*num_train_steps/BATCH_SIZE)

optimizer = optimization.create_optimizer(init_lr=LEARNING_RATE,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')

# compiling the model
model.compile(optimizer=optimizer,
              loss=loss,
              metrics=['accuracy'])

history = model.fit(x=train_X,
                    y=train_y,
                    validation_data=(val_X, val_y),
                    epochs= 5,
                    validation_steps=1,
                    verbose=1,
                    batch_size= 32)

# Precision , Recall , F1-score
cr = classification_report(train_y_labels, preds_labels, target_names = ['Negative', 'Neutral', 'Positive'])
print(cr)

