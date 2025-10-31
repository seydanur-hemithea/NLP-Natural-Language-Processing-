# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 17:07:04 2025

@author: asus
"""

import spacy 
nlp=spacy.load("en_core_web_sm")

sentence1="What is the weather like today or tomorrow"
doc1=nlp(sentence1)
for token in doc1:
    print(token.text,token.pos_)
"""  What PRON
    is AUX
    the DET
    weather NOUN
    like ADP
    today NOUN
    or CCONJ
    tomorrow NOUN"""