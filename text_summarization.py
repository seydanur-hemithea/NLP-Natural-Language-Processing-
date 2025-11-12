# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 16:23:58 2025

@author: asus
"""
from transformers import pipeline

summarizer=pipeline("summarization")


text="""Machine learning is a branch of artificial intelligence
 that enables computers to learn from data without being explicitly 
 programmed. Instead of following fixed rules, a machine learning model
 identifies patterns, adapts to new information, and improves its performance
 over time. It’s like training an actor to improvise—given enough rehearsal (data),
 the model begins to predict, classify, or generate outcomes with increasing 
 accuracy. From spam filters to recommendation systems, machine learning powers 
 many of the intelligent tools we use every day.
"""
summary=summarizer(
   text,
   max_length=50,
   min_length=10,
   do_sample=True)

print(summary[0]["summary_text"])



