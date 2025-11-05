# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 20:58:39 2025

@author: asus
"""

from transformers import BertTokenizer,BertModel

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

model_name="bert-base-uncased"
tokenizer=BertTokenizer.from_pretrained(model_name)
model=BertModel.from_pretrained(model_name)
documents=["Machine Learning is a field of artificial intelligence",
           "Natural Language processing involves understanding human language",
           "Artificial intelligence encomppases machine learning and natural language processing(NLP)",
           "Deep Learning is a subset of machine learning",
           "Data science combines statistics,data analysis and machine learning",
           "I go to shop"
           ]

Query="What is Deep Learning?"

def get_embedding(text):
    
    inputs=tokenizer(text,return_tensors="pt",truncation=True,padding=True)
    
    outputs=model(**inputs)
    last_hidden_state=outputs.last_hidden_state
    embedding=last_hidden_state.mean(dim=1)
    return embedding.detach().numpy()

doc_embeddings=np.vstack([get_embedding(doc)for doc in documents])
Query_embedding=get_embedding(Query)
#cos similarity
similarities=cosine_similarity(Query_embedding,doc_embeddings)
#similarrity score per documents
for i,score in enumerate(similarities[0]):
    print(f"Document:{documents[i]} \n{score}")
    
    

"""
Document:Machine Learning is a field of artificial intelligence 
0.634821891784668
Document:Natural Language processing involves understanding human language 
0.626939058303833
Document:Artificial intelligence encomppases machine learning and natural language processing(NLP) 
0.5046247243881226
Document:Deep Learning is a subset of machine learning 
0.6263622045516968
Document:Data science combines statistics,data analysis and machine learning 
0.5866265892982483
Document:I go to shop 
0.5354946255683899"""

most_similar_index=similarities.argmax()
print=(f"Most Similar Index:{documents[most_similar_index]}")


