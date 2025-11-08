# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 16:10:24 2025

@author: asus
"""
from transformers import MarianMTModel,MarianTokenizer

model_name="Helsinki-NLP/opus-mt-en-fr"
tokenizer=MarianTokenizer.from_pretrained(model_name)
model=MarianTokenizer.from_pretrained(model_name)

text="Hello,what is your name?"
#encoding and giving as input 
translated_text=model.generate(**tokenizer(text,return_tensors="pt",padding=True))

translated_text=tokenizer.decode(translated_text[0],skip_special_tokens=True)
print(f"translated_text:{translated_text}")
