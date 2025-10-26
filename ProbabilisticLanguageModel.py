# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 16:22:59 2025

@author: asus
"""

import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

from collections import Counter

corpus= [
"I love reading books at night",
"I love watching movies on weekends",
" They love watching movies at home",
"I love listening to music while working",
"She enjoys reading books every evening",
"She enjoys watching movies with friends",
"He likes reading books before be",
"He likes watching movies after dinner",
"They love reading books together",
"I love reading books with my sister"]


##N-gram

tokens=[word_tokenize(sentence.lower()) for sentence in corpus]

bigram=[]
for token_list in tokens:
    bigram.extend(list(ngrams(token_list,2)))
bigram_freq=Counter(bigram)


trigram=[]
for token_list in tokens:
    trigram.extend(list(ngrams(token_list,3)))
trigram_freq=Counter(trigram)

bigram=("love","reading")#target bigram

#"love reading book" statistical  
prob_books=trigram_freq[("love","reading","books")]/bigram_freq[bigram]
print(f"books probability:{prob_books}")

bigram=("love","watching")#target bigram

  
prob_movies=trigram_freq[("love","watching","movies")]/bigram_freq[bigram]
print(f"movies probability:{prob_movies}")
#%%
#Hidden markow model


import nltk
from nltk.tag import hmm


train_data=[[("I","PRP"),("am","VBP"),("a","DT"),("teacher","NN")],
            [("You","PRP"),("are","VBP"),("a","DT"),("student","NN")]]

trainer=hmm.HiddenMarkovModelTrainer()
hmm_tagger=trainer.train(train_data)

test_sentence="I am a student".split()
tags=hmm_tagger.tag(test_sentence)
print(f"test sentence:{tags}")
"""
test sentence:[('I', 'PRP'), ('am', 'VBP'), ('a', 'DT'), ('student', 'NN')]
"""
test_sentence="He is a driver".split()
tags=hmm_tagger.tag(test_sentence)
print(f"test sentence:{tags}")


"""
test sentence:[('He', 'PRP'), ('is', 'PRP'), ('a', 'PRP'), ('driver', 'PRP')]
"""


#%%max entropy

from nltk.classify import MaxentClassifier


train_data=[({"love":True,"amazing":True,"happy":True,"terrible":False},"positive"),
            ({"hate":True,"terrible":True},"negative"),
            ({"joy":True,"happy":True,"hate":False},"positive"),
            ({"sad":True,"depressed":True,"love":False},"negative")]
classifier=MaxentClassifier.train(train_data,max_iter=10)

test_sentence="I hate this movie and it was terrible"
features={word:(word in test_sentence.lower().split()) for word in ["love","amazing","terrible","happy","joy","sad","depressed","hate"]}

label=classifier.classify(features)
print(f"result:{label}")







