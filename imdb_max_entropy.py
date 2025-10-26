# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 16:30:12 2025

@author: asus
"""

import pandas as pd
import re
import matplotlib.pyplot as plt
import nltk
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from nltk.classify import MaxentClassifier



dataset= pd.read_csv("IMDB_Dataset.csv")

documents=dataset['review']
labels=['sentiment'] 
nltk.download("stopwords")

stopwords_eng=set(stopwords.words("english"))

def clean_text(text):
    text=text.lower()
    text=re.sub(r"\d+","",text)
    text=re.sub(r"[^\w\s]","",text)
    word_token=nltk.word_tokenize(text)    
        
    filtered_words = [word for word in word_token if word.lower() not in stopwords_eng]
    return " ".join(filtered_words)

  
cleaned_documents=[clean_text(doc) for doc in documents]
train_data=[({"love":True,"amazing":True,"happy":True,"terrible":False},"positive"),
            ({"hate":True,"terrible":True},"negative"),
            ({"joy":True,"happy":True,"hate":False},"positive"),
            ({"sad":True,"depressed":True,"love":False},"negative")]
classifier=MaxentClassifier.train(train_data,max_iter=10)

for i,sentences in enumerate(cleaned_documents):
    features={word:(word in sentences.lower().split()) for word in ["love","amazing","terrible","happy","joy","sad","depressed","hate"]}
    
    label=classifier.classify(features)
    print(f"Index: {i} → Sentiment: {label}")








