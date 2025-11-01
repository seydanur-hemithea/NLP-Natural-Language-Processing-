# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 21:30:46 2025

@author: asus
"""
#https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/amazon.csv

import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
nltk.download("vader_lexicon")
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")





import csv
import string
dataset = pd.read_csv(
    "amazon_dataset.csv")
dataset["sentiment"] = dataset["reviewText"].astype(str).str[-1:]

def clean_preprocess_data(text):
    if not isinstance(text, str):
        return ""
    
    # Küçük harfe çevir
    text = text.lower()
    
    # Noktalama işaretlerini kaldır
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Sayıları kaldır
    text = re.sub(r"\d+", "", text)
    
    # Tokenize et
    tokens = word_tokenize(text)
    
    # Stopword'leri çıkar
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize et
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    
    # Tekrar birleştir
    processed_text = " ".join(lemmatized_tokens)
    
    return processed_text
dataset["reviewText2"]=dataset["reviewText"].apply(clean_preprocess_data)


analyzer=SentimentIntensityAnalyzer()
def get_sentiments(text):
    score=analyzer.polarity_scores(text)
    sentiment_pred=1 if score["pos"]>0 else 0
    return sentiment_pred
dataset["sentiment_pred"]=dataset["reviewText2"].apply(get_sentiments)
    


dataset["sentiment"] = dataset["sentiment"].apply(lambda x: int(x) if x in ["0", "1"] else None)
dataset = dataset.dropna(subset=["sentiment"]) 

from sklearn.metrics import confusion_matrix,classification_report
cm=confusion_matrix(dataset["sentiment"],dataset["sentiment_pred"])
print(f"Confusion matrix:{cm}")

cr=classification_report(dataset["sentiment"],dataset["sentiment_pred"])
print(f"Classification report:{cr}")



"""Confusion matrix:[[ 1148  3619]
 [  602 14631]]
Classification report:              precision    recall  f1-score   support

           0       0.66      0.24      0.35      4767
           1       0.80      0.96      0.87     15233

    accuracy                           0.79     20000
   macro avg       0.73      0.60      0.61     20000
weighted avg       0.77      0.79      0.75     20000
"""









