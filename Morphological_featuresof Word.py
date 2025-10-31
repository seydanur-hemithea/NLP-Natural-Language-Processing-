# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 16:29:13 2025

@author: asus
"""
import spacy
nlp=spacy.load("en_core_web_sm")
#python -m spacy download en_core_web_sm 
word="I go to school"
doc=nlp(word)

for token in doc:
    
    print(f"text: {token.text}")  # original word
    print(f"lemma: {token.lemma_}")  # base form
    print(f"POS: {token.pos_}")  # part of speech
    print(f"Tag: {token.tag_}")  # detailed grammatical tag
    print(f"Dependency: {token.dep_}")  # syntactic role in sentence
    print(f"Shape: {token.shape_}")  # character pattern (e.g., Xxxx, dddd)
    print(f"Is alpha: {token.is_alpha}")  # is the token alphabetic?
    print(f"Is stop: {token.is_stop}")  # is the token a stop word?
    print(f"Morphology: {token.morph}")  # morphological features

    print(f"Is plural: {'Number=Plur' in token.morph}")


    
""" text: I
    lemma: I
    POS: PRON
    Tag: PRP
    Dependency: nsubj
    Shape: X
    Is alpha: True
    Is stop: True
    Morphology: Case=Nom|Number=Sing|Person=1|PronType=Prs
    Is plural: False
    text: go
    lemma: go
    POS: VERB
    Tag: VBP
    Dependency: ROOT
    Shape: xx
    Is alpha: True
    Is stop: True
    Morphology: Tense=Pres|VerbForm=Fin
    Is plural: False
    text: to
    lemma: to
    POS: ADP
    Tag: IN
    Dependency: prep
    Shape: xx
    Is alpha: True
    Is stop: True
    Morphology: 
    Is plural: False
    text: school
    lemma: school
    POS: NOUN
    Tag: NN
    Dependency: pobj
    Shape: xxxx
    Is alpha: True
    Is stop: False
    Morphology: Number=Sing
    Is plural: False"""
