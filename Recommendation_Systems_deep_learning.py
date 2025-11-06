# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 21:45:37 2025

@author: asus
"""
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense,Input,Embedding,Flatten,Dot,Dense
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model

import warnings
warnings.filterwarnings("ignore")

user_ids=np.array([0,1,2,3,4,0,1,2,3,4])
item_ids=np.array([0,1,2,3,4,1,2,3,4,5])
ratings=np.array([5,4,3,2,1,4,5,3,2,1,])

user_ids_train,user_ids_test,item_ids_train,item_ids_test=train_test_split(user_ids,item_ids,test_size=0.2,random_state=42)
ratings_train, ratings_test = train_test_split(ratings, test_size=0.2, random_state=42)
#create neural network

def create_model(num_users,num_items,embedding_dim):
    #input layer
    user_input=Input(shape=(1,),name="user")
    item_input=Input(shape=(1,),name="item")
    #embedding layer
    user_embedding=Embedding(input_dim=num_users,output_dim=embedding_dim,name="user_embedding")(user_input)
    item_embedding=Embedding(input_dim=num_items,output_dim=embedding_dim,name="item_embedding")(item_input)
    
    #flattening vectors
    
    user_vecs=Flatten()(user_embedding)
    item_vecs=Flatten()(item_embedding)
    
    dot_product=Dot(axes=1)([user_vecs,item_vecs])
    output = Dense(1)(dot_product)
    
    model=Model(inputs=[user_input,item_input],outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001),loss="mean_squared_error")
    
    return model
    

#train and test
num_users=5
num_items=6
embedding_dim=8
model=create_model(num_users,num_items,embedding_dim)
model.fit([user_ids_train,item_ids_train],ratings_train,epochs=100,verbose=1,validation_split=0.1)
    
loss=model.evaluate([user_ids_test,item_ids_test],ratings_test)
print(f"test:{loss}")

user_id=np.array([0])
item_id=np.array([5])
prediction=model.predict([user_id,item_id])

print(f"Predicted arting for user:{user_id[0]},item:{item_id[0]},{prediction[0][0]:.2}")
    
    
    
    
    
    
    
    
    
