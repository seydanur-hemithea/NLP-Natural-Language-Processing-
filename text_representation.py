# -*- coding: utf-8 -*-

"""
#Feature extraction
#For the computer to understand
#model training 

#Algorithms
1.bag of words :word frequency
2. TF-IDF (Term Frequency – Inverse Document Frequency:Word frequency and its importance across all documents"
3. Word Embeddings (Word2Vec, GloVe, FastText
4. Contextual Embeddings (BERT, RoBERTa, GPt   """""   
                          

#%%
#method1 
#count vektorizer bag of words
from sklearn.feature_extraction.text import CountVectorizer
#create dataset
dokuments=["cat eat",
           "cat run"]
#definate vektorizer
vectorizer=CountVectorizer()
#text translate numeric vektor
X=vectorizer.fit_transform(dokuments)
#results examining and vector representation
feature_names=vectorizer.get_feature_names_out()#word cluster
vector_representation=X.toarray()

#%%method 2  TF-IDF 
import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


documents=["the dog is sweet animal" ,"dogs and births are  sweet animals ",
           "cows produce milk"]

tfidf_vectorizer=TfidfVectorizer()
X=tfidf_vectorizer.fit_transform(documents)
feature_names2=tfidf_vectorizer.get_feature_names_out()

vector_represantation2=X.toarray()
print(f"tfidf:{vector_represantation2}")
df_tfidf=pd.DataFrame(vector_represantation2,columns=feature_names2)
tf_idf=df_tfidf.mean(axis=0)
"""
- High score = The word appears frequently in the document and is generally rare across the corpus.
- Low score = The word is either very common or appears infrequently in that specific document
"""
#%%N-Gram
"""
- n = 1 → unigram 
- n = 2 → bigram 
- n = 3 → trigram 
"""
from sklearn.feature_extraction.text import CountVectorizer
documents=["This work is NGram algorthm",
"This work is natural language processing algorithm"]

vectorizer_unigram=CountVectorizer(ngram_range=(1,1))
vectorizer_bigram=CountVectorizer(ngram_range=(2,2))
vectorizer_trigram=CountVectorizer(ngram_range=(3,3))

X_unigram=vectorizer_unigram.fit_transform(documents)
unigram_features=vectorizer_unigram.get_feature_names_out()

X_bigram=vectorizer_bigram.fit_transform(documents)
bigram_features=vectorizer_bigram.get_feature_names_out()


X_trigram=vectorizer_trigram.fit_transform(documents)
trigram_features=vectorizer_trigram.get_feature_names_out()

#%%word embedding
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.models import Word2Vec,FastText
from gensim.utils import simple_preprocess#cleaning and tokenize

sentences=[
    " The sun was shining brightly as children played in the summer breeze",
    "She decided to write a novel about forgotten memories and silent voices",
    "The apple fell from the tree and rolled gently down the hill",
    "He opened the book and began to read under the flickering candlelight",
    "During the summer, they traveled across Europe and visited ancient ruins"]
tokenize_sent=[simple_preprocess(sentence)for sentence in sentences ]

word2vec_model=Word2Vec(sentences=tokenize_sent,vector_size=50,window=5,min_count=1,sg=0)
"""- sentences: This is the input data, consisting of tokenized sentences. Each sentence is a list of words that the model will learn from.
- vector_size: This defines the number of dimensions in the word vectors. A higher value captures more semantic detail. In our case, we use 50 dimensions.
- window: This sets the maximum distance between the target word and its surrounding context words. A window size of 5 means the model looks at 5 words before and after the target word.
- min_count: This filters out words that appear less than the specified number of times. Setting it to 1 ensures that all words are included in the training.
- sg: This determines the training algorithm. If set to 0, the model uses CBOW (Continuous Bag of Words), which predicts a word based on its context. If set to 1, it uses Skip-gram, which predicts context words from a target word. In this setup, we use CBOW.
"""
fastText_model=FastText(sentences=tokenize_sent,vector_size=50,window=5,min_count=1,sg=0)



def plot_word_embedding(model,title):
    word_vectors=model.wv
    words=list(word_vectors.index_to_key)[:1000]
    vectors=[word_vectors[word] for word in words]
    
    
    pca=PCA(n_components=3)
    reduced_vectors=pca.fit_transform(vectors)
    
    fig=plt.figure(figsize=(8,6))
    ax=fig.add_subplot(111,projection="3d")
    
    ax.scatter(reduced_vectors[:,0],reduced_vectors[:,1],reduced_vectors[:,2])
    
    for i,word in enumerate(words):
        ax.text(reduced_vectors[i,0],reduced_vectors[i,1],reduced_vectors[i,2],word,fontsize=12)
    ax.set_title(title)
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.set_zlabel("component 3")
    plt.show()

plot_word_embedding(word2vec_model,"word2vector")

plot_word_embedding(fastText_model,"fasttext")

#%% transformers_based_text_representation
import torch
from transformers import AutoTokenizer,AutoModel


model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

#pretrained model with BERT

text= "Transformers can be used for natural language processing"

inputs=tokenizer(text,return_tensors="pt")#“The output is returned as a PyTorch tensor.”

with torch.no_grad():
    ## Gradient calculation is disabled to use memory more efficiently    
    outputs=model(**inputs)
last_hidden_state=outputs.last_hidden_state# include all tokens outputs
first_token_embedding=last_hidden_state[0,0,:].numpy()

print(f"text_represantation:{first_token_embedding}")




























