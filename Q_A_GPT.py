# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 17:08:11 2025

@author: asus
"""

from transformers import GPT2Tokenizer, GPT2LMHeadModel

import torch

import warnings
warnings.filterwarnings("ignore")


model_name="gpt2"

tokenizer=GPT2Tokenizer.from_pretrained(model_name)

model = GPT2LMHeadModel.from_pretrained(model_name)
def generate_answer(context, question):
    input_text = f"Question: {question}\nContext: {context}\nAnswer:"
    inputs=tokenizer.encode(input_text,return_tensors="pt")
    
    with torch.no_grad():
        outputs=model.generate(inputs,max_length=500)
    
    answer=tokenizer.decode(outputs[0],skip_special_tokens=True)
    answer=answer.split("Answer:")[-1].strip()
    return answer
question = "Where do the bears live?"
context = "Extant bears are found in sixty countries primarily in the Northern Hemisphere and are concentrated in Asia, North America, and Europe. An exception is the spectacled bear; native to South America, it inhabits the Andean region.[58] The sun bear's range extends below the equator in Southeast Asia.[59] The Atlas bear, a subspecies of the brown bear was distributed in North Africa from Morocco to Libya, but it became extinct around the 1870s"
answer = generate_answer(context, question)
print(f"Answer: {answer}")



