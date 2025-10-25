# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 17:08:02 2025

@author: şeydanur
"""
import pandas as pd
import re
import matplotlib.pyplot as plt
import nltk

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from gensim.models import Word2Vec,FastText
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords


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
tokenized_doc=[simple_preprocess(doc) for doc in cleaned_documents]


word2vec_model=Word2Vec(sentences=tokenized_doc,vector_size=50,window=5,min_count=1,sg=0)
word_vectors=word2vec_model.wv

words=list(word_vectors.index_to_key[:500])
vectors=[word_vectors[word] for word in words]

kmeans=KMeans(n_clusters=2)
kmeans.fit(vectors)
clusters=kmeans.labels_
#K-Means is used with Word2Vec to group semantically similar word vectors into meaningful clusters for unsupervised analysis.



pca=PCA(n_components=2)#for visualition
reduced_vectors=pca.fit_transform(vectors)

plt.figure()
plt.scatter(reduced_vectors[:,0],reduced_vectors[:,1],c=clusters,cmap="viridis")
centers=pca.transform(kmeans.cluster_centers_)
plt.scatter(centers[:,0],centers[:,1],c="red",marker="x",s=150,label="center")
plt.legend()

for i,word in enumerate(words):
    plt.text(reduced_vectors[i,0],reduced_vectors[i,1],word,fontsize=7)
    
plt.title("Word2Vec")



# FastText model
fasttext_model = FastText(sentences=tokenized_doc, vector_size=50, window=5, min_count=1, sg=0)
fasttext_vectors = fasttext_model.wv


fasttext_words = list(fasttext_vectors.index_to_key[:500])
fasttext_vecs = [fasttext_vectors[word] for word in fasttext_words]


fasttext_kmeans = KMeans(n_clusters=2)
fasttext_kmeans.fit(fasttext_vecs)
fasttext_clusters = fasttext_kmeans.labels_


fasttext_pca = PCA(n_components=2)
fasttext_reduced = fasttext_pca.fit_transform(fasttext_vecs)
fasttext_centers = fasttext_pca.transform(fasttext_kmeans.cluster_centers_)


plt.figure()
plt.scatter(fasttext_reduced[:,0], fasttext_reduced[:,1], c=fasttext_clusters, cmap="plasma")
plt.scatter(fasttext_centers[:,0], fasttext_centers[:,1], c="blue", marker="x", s=150, label="center")
plt.legend()

for i, word in enumerate(fasttext_words):
    plt.text(fasttext_reduced[i,0], fasttext_reduced[i,1], word, fontsize=7)

plt.title("FastText")






