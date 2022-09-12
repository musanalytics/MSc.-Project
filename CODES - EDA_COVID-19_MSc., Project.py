#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: hunchoahmad
"""

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

import warnings
%matplotlib inline
warnings.filterwarnings('ignore')

#reading and understanding the data
train = pd.read_csv("/Users/hunchoahmad/Downloads/Corona_NLP_train.csv", encoding="latin-1") 
test = pd.read_csv("/Users/hunchoahmad/Downloads/Corona_NLP_test.csv", encoding="latin-1")
train.head()

print(train.info(),'\n')
test.info()


train.drop_duplicates(inplace=True)
test.drop_duplicates(inplace=True)

#Finding locations
train['Location'].value_counts()[:60]

train.drop(['UserName', 'ScreenName'], axis=1, inplace=True)
test.drop(['UserName', 'ScreenName'], axis=1, inplace=True)

# Merging the two datasets
combined = pd.concat([train, test], ignore_index= True)

#Mentions
mentions = train['OriginalTweet'].str.extractall(r"(@\S+)")
mentions = mentions[0].value_counts()
mentions[:50]

# Selecting relevant features: tweet and Sentiments
combined = combined.loc[:, ["OriginalTweet", "Sentiment"]]

# load stop words
stop_word = stopwords.words('english')

#dates when tweets were made
train['TweetAt'].value_counts()

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
train.Location = train.Location.str.split(',').str[0]

#Plotting sentiments

labels = ['Positve', 'Negative', 'Neutral', 'Extremely Positive', 'Extremely Negative']
colors = ['lightblue','lightsteelblue','silver', 'deepskyblue', 'skyblue']
explode = (0.1, 0.1, 0.1, 0.1, 0.1)
plt.pie(train.Sentiment.value_counts(), colors = colors, labels=labels,
        shadow=300, autopct='%1.1f%%', startangle=90, explode = explode)
plt.show()

# replace "extremely positive/negative" with "postive/negative"
train["Sentiment"] = train["Sentiment"].str.replace("Extremely Negative", "Negative")
train["Sentiment"] = train["Sentiment"].str.replace("Extremely Positive", "Positive")

test['Sentiment'] = test.Sentiment.str.replace('Extremely Positive', 'Positive')
test['Sentiment'] = test.Sentiment.str.replace('Extremely Negative', 'Negative')
#Narrowed Sentiments 

labels = ['Positve', 'Negative', 'Neutral']
colors = ['lightblue','lightsteelblue','silver', 'deepskyblue', 'skyblue']
explode = (0.1, 0.1, 0.1, 0.1, 0.1)
plt.pie(train.Sentiment.value_counts(), colors = colors, labels=labels,
        shadow=300, autopct='%1.1f%%', startangle=90, explode = explode)
plt.show()

#Plotting twitter icon wordclouds
mask1 = np.array(Image.open('twitter_icon_1.jpeg'))

from PIL import image
#Neutral Tweets
wc=WordCloud(max_words=700,mask=mask1,background_color='white').generate(' '.join(Neutral)) 
plt.figure(figsize=(7,7))
plt.axis('off')
plt.imshow(wc)

#Negative Tweets
wc=WordCloud(max_words=700,mask=mask1,background_color='white').generate(' '.join(Negative)) 
plt.figure(figsize=(7,7))
plt.axis('off')
plt.imshow(wc)

#Positive Tweets
wc=WordCloud(max_words=700,mask=mask1,background_color='white').generate(' '.join(Positive)) 
plt.figure(figsize=(7,7))
plt.axis('off')
plt.imshow(wc)

#Heatmap using Folium  (Use of more coordinates have been made)
import folium
from folium.plugins import HeatMap

mapObj = folium.Map(location = [51.49799827422944, -0.13568476148837225], zoom_start=10)
mapObj.save("output.html")

data = [

        [40.749990744243966, -73.98561591917483, 0.50],
        [40.749990744243966, -73.98561591917483, 0.50],
        [28.530954150187892, -81.37158323971082, 0.50],
        [28.530954150187892, -81.37158323971082, 0.50],
        [40.78656614403477, -73.19386240332194, 0.50], 
        [34.028129, -118.262080, 0.50],
        [-26.108076700695474, 28.050298311478343, 0.50], 
        [-26.108076700695474, 28.050298311478343, 0.50],
        [19.076195, 72.869331, 0.50], 
        [19.076195, 72.869331, 0.50], 
        [43.69333693528487, -79.34442026452643, 0.50],
        [43.69333693528487, -79.34442026452643, 0.50], 
        [45.52883296949091, -73.53394076865982, 0.50],
        [38.93641192631958, -77.06814283353853, 0.50],
        [38.93641192631958, -77.06814283353853, 0.50], 
        [38.93641192631958, -77.06814283353853, 0.50],
        [53.43891823506528, -6.254636893944279, 0.50],
        [51.575060195248305, -0.15950296941519643, 0.50], 
        [53.43891823506528, -6.254636893944279, 0.50],
        [51.50548473329821, -0.12563792881243846 , 0.50],
        [51.50548473329821, -0.12563792881243846 , 0.50],
        [41.305197386612875, 1.9835637307198177, 0.50],
        [41.305197386612875, 1.9835637307198177, 0.50], 
        [28.68341385249912, 77.20828880316458, 0.50],
        [28.68341385249912, 77.20828880316458, 0.50], 
        [28.68341385249912, 77.20828880316458, 0.50], 
        [47.603411628449834, -122.32015449499292, 0.50],
        [47.603411628449834, -122.32015449499292, 0.50],
        [42.32265346527708, -87.64410554082671, 0.50],
        [42.32265346527708, -87.64410554082671, 0.50],
        [30.734472443521526, 76.78996244672244, 0.50],
        [38.520370689897895, -98.21231602445602, 0.50],
        [38.520370689897895, -98.21231602445602, 0.50],
        [6.526205902003814, 3.3745857568493185, 0.50],
        [6.526205902003814, 3.3745857568493185, 0.50], 
        [9.077479911921184, 7.395620339753439, 0.50],
        [9.077479911921184, 7.395620339753439, 0.50], 
        [31.56013515357674, -99.29636392600847, 0.50],
        [31.56013515357674, -99.29636392600847, 0.50],
        [31.56013515357674, -99.29636392600847, 0.50],
        [-33.83652083112268, 151.18599717943437, 0.50],
        [-33.83652083112268, 151.18599717943437, 0.50],
        [-33.83652083112268, 151.18599717943437, 0.50], 
        [-37.804398027362204, 144.99702576190614, 0.50], 
        [-37.804398027362204, 144.99702576190614, 0.50], 
        [34.059990340155444, -118.2277478274139, 0.50],
        [34.059990340155444, -118.2277478274139, 0.50],
        [26.0315278630918, -80.3005967285992, 0.50],
        [26.0315278630918, -80.3005967285992, 0.50]
        ]



