# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 17:21:37 2025

@author: asus
"""

import nltk
from nltk.wsd import lesk

s1="I go to the bank to deposit money"
w1="bank"
sense1=lesk(nltk.word_tokenize(s1),w1)
print(f"Sentence1:{s1}")
print(f"word1:{w1}")
print(f"sense1:{sense1.definition()}")
s2="The bank of  river is flooded after the heavy rain"
w2="bank"
sense2=lesk(nltk.word_tokenize(s2),w2)
print(f"Sentence2:{s2}")
print(f"word2:{w2}")
print(f"sense2:{sense2.definition()}")

"""Sentence1:I go to the bank to deposit money
word1:bank
sense1:a container (usually with a slot in the top) for keeping money at home
Sentence2:The bank of river is flooded after the heavy rain
word2:bank
sense2:a slope in the turn of a road or track; the outside is higher than the inside in order to reduce the effects of centrifugal force"""
#%%

from pywsd.lesk import simple_lesk,adapted_lesk,cosine_lesk

import nltk
nltk.download('averaged_perceptron_tagger_eng')

sentences=["I go to the bank to deposit money",
"The river bank is flooded after the heavy rain"]

word="bank"

for s in sentences:
    print(f"Sentence: {s}")
    
    sense_simple_lesk = simple_lesk(s, word)
    if sense_simple_lesk:
        print(f"Sense Simple: {sense_simple_lesk.definition()}")
    else:
        print("Sense Simple: No definition found.")
    
    sense_adapted_lesk = adapted_lesk(s, word)
    if sense_adapted_lesk:
        print(f"Sense Adapted: {sense_adapted_lesk.definition()}")
    else:
        print("Sense Adapted: No definition found.")
    
    sense_cosine_lesk = cosine_lesk(s, word)
    if sense_cosine_lesk:
        print(f"Sense Cosine: {sense_cosine_lesk.definition()}")
    else:
        print("Sense Cosine: No definition found.")
    
  
    
"""  
Sentence: I go to the bank to deposit money
Sense Simple: a financial institution that accepts deposits and channels the money into lending activities
Sense Adapted: a financial institution that accepts deposits and channels the money into lending activities
Sense Cosine: a container (usually with a slot in the top) for keeping money at home

Sentence: The river bank is flooded after the heavy rain
Sense Simple: sloping land (especially the slope beside a body of water)
Sense Adapted: sloping land (especially the slope beside a body of water)
Sense Cosine: a supply or stock held in reserve for future use (especially in emergencies)
"""

















