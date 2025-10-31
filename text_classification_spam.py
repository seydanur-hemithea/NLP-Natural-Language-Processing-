# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 16:38:11 2025

@author: asus
"""

import pandas as pd
import nltk 
import re
from nltk.corpus import stopwords#stopwords
from nltk.stem import WordNetLemmatizer#lemmatization


data=pd.read_csv("spam.csv",encoding="latin-1")
data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1,inplace=True)
data.columns=["label","text"]

print(data.isna().sum()) #is there missing value?


nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


text=list(data.text)
lemmatizer=WordNetLemmatizer()

corpus=[]
for i in range(len(text)):
    r=re.sub("[^a-zA-Z]"," ",text[i])
    r=r.lower()
    r=r.split()
    r=[word for word in r if word not in stopwords.words("english")]
    r=[lemmatizer.lemmatize(word)for word in r]
    r=" ".join(list(r))
    corpus.append(r)
    
data["text2"]=corpus   


#%%model training and evaluation
X=data["text2"]
y=data["label"]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#feature extraction:bag of words

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X_train_cv=cv.fit_transform(X_train)

#classifier training:model training and evaluation
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(X_train_cv,y_train)

X_test_cv=cv.transform(X_test)
#prediction

prediction=dt.predict(X_test_cv)
from sklearn.metrics import confusion_matrix
c_matrix=confusion_matrix(y_test, prediction)

accuracy=100*sum(sum((c_matrix))-c_matrix[1,0]-c_matrix[0,1])/sum(sum(c_matrix))

print(accuracy)
"""
true_positive = c_matrix[0,0]
true_negative = c_matrix[1,1]
total = c_matrix.sum()
accuracy = 100 * (true_positive + true_negative) / total
print(f"Accuracy: {accuracy:.2f}%")"""












