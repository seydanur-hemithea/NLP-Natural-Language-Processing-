# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 22:28:58 2025

@author: asus
"""
import string
import re
import nltk#TextBlob, sahneye çıkarken perde arkasında NLTK’nin punkt veri paketini kullanır—özellikle cümle ve kelime ayırma (tokenization) görevlerinde

nltk.download('punkt')#metni kelime ve cümle bazında tokenlara ayrır
from textblob import TextBlob#pip install textblob,metin analizlerinde kullanilir
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer# for stemming funtion
from nltk.stem import WordNetLemmatizer#for lemmatizer function
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


df = pd.read_csv("spam.csv", encoding="latin1", usecols=[0, 1], names=["label", "message"], skiprows=1)
documents=df["message"]

def Cleaned_text(text):
    text=" ".join(text.split())
    text=text.lower()
    text=text.translate(str.maketrans("","",string.punctuation))
    
    
    word_token=nltk.word_tokenize(text)
    nltk.download("stopwords")

    stopwords_eng=set(stopwords.words("english"))
    filtered_words = [word for word in word_token if word.lower() not in stopwords_eng]
    return " ".join(filtered_words)
cleaned_doc=[Cleaned_text(message) for message in documents] 


vectorizer=TfidfVectorizer()
cleaned_doc_str = cleaned_doc 
X=vectorizer.fit_transform(cleaned_doc_str)

feature_names=vectorizer.get_feature_names_out()
tf_idf_score=X.mean(axis=0).A1

df_tfidf=pd.DataFrame({"word":feature_names,"tfidf_score":tf_idf_score})
df_tfidf_sorted=df_tfidf.sort_values(by="tfidf_score",ascending=False)
print(df_tfidf_sorted.head(10))
