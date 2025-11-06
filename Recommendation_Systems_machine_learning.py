# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 17:39:02 2025

@author: asus
"""

from surprise import Dataset,KNNBasic,accuracy
from surprise.model_selection import train_test_split

data=Dataset.load_builtin("ml-100k") #user id,fil id,score


trainset,testset=train_test_split(data,test_size=0.2)

#ml model create:KNN

model_options={"name":"cosine",
               "user_based":True#similiarity  between users
               }
#train
model=KNNBasic(sim_options=model_options)
model.fit(trainset)
#test
prediction=model.test(testset)
accuracy.rmse(prediction)
#recommandation system
def get_top_n(predictions,n=10):
    top_n={}
    
    for uid,iid ,true_r,est,_ in predictions:
        if not top_n.get(uid):
            top_n[uid]=[]
        top_n[uid].append((iid,est))
    for uid,user_ratings in top_n.items():
        user_ratings.sort(key=lambda x:x[1],reverse=True)
        top_n[uid]=user_ratings[:n]
        
    return top_n
n=5
top_n=get_top_n(prediction,n)
user_id="3"
print(f"top  {n} recommandation for user {user_id}")
for item_id,rating in top_n[user_id]:
    print(f"item id:{item_id},score:{rating}")
    