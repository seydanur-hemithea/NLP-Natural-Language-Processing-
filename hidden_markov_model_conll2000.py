# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 22:34:45 2025

@author: asus
"""

import nltk 
from nltk.tag import hmm
from nltk.corpus import  conll2000#for POS( Part Of Speech) dataset

nltk.download("conll2000")
train_data=conll2000.tagged_sents("train.txt")
test_data=conll2000.tagged_sents("test.txt")

trainer=hmm.HiddenMarkovModelTrainer()
hmm_tagger=trainer.train(train_data)
 
test_sentence="I like going to school".split()
tags=hmm_tagger.tag(test_sentence)
print(f"test sentence:{tags}")
"""
test sentence:[('I', 'PRP'), ('like', 'IN'), ('going', 'VBG'), ('to', 'TO'), ('school', 'NN')]
"""







