# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 21:50:25 2025

@author: asus
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
from collections import Counter
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords")
stopwords_eng=set(stopwords.words("english"))
nltk.download("wordnet")
nltk.download('punkt_tab')





dataset= pd.read_csv("IMDB_Dataset.csv")


documents=dataset['review']
labels=['sentiment']

def clean_text(text):
    text=text.lower()
    text=re.sub(r"\d+","",text)
    text=re.sub(r"[^\w\s]","",text)
    text=" ".join([word for word in text.split() if len(word)>2])
    word_token=nltk.word_tokenize(text)            
    
    
    text=[word for word in word_token if word.lower() not in stopwords_eng]
    print(text)
    return text
cleaned_doc=[clean_text(row) for row in documents] 
#%%
vectorizer=CountVectorizer() 
cleaned_doc_str = [" ".join(doc) for doc in cleaned_doc]
X = vectorizer.fit_transform(cleaned_doc_str[:75])
feature_names=vectorizer.get_feature_names_out()
vector_representation=X.toarray()
print(f"vector representation: {vector_representation}")
df_bow=pd.DataFrame(vector_representation,columns=feature_names)
word_count=X.sum(axis=0).A1
word_freq=dict(zip(feature_names,word_count))


most_common_5_words=Counter(word_freq).most_common(5)

print(f" most commen 5 words:{most_common_5_words}")










 