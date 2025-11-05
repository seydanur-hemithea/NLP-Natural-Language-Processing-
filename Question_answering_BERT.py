# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 16:15:15 2025

@author: asus
"""

#question answering
from transformers import BertTokenizer, BertForQuestionAnswering
import torch

import warnings
warnings.filterwarnings("ignore")
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
tokenizer=BertTokenizer.from_pretrained(model_name)
model=BertForQuestionAnswering.from_pretrained(model_name)


def predict_answer(context, question):
    encoding = tokenizer.encode_plus(question, context, return_tensors="pt", max_length=512, truncation=True)
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    with torch.no_grad():
        start_scores, end_scores = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)

    start_index = torch.argmax(start_scores, dim=1).item()
    end_index = torch.argmax(end_scores, dim=1).item()

    answer_tokens = tokenizer.convert_ids_to_tokens(input_ids[0][start_index:end_index+1])
    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    return answer
question = "Where do the bears live?"
context = "Extant bears are found in sixty countries primarily in the Northern Hemisphere and are concentrated in Asia, North America, and Europe. An exception is the spectacled bear; native to South America, it inhabits the Andean region.[58] The sun bear's range extends below the equator in Southeast Asia.[59] The Atlas bear, a subspecies of the brown bear was distributed in North Africa from Morocco to Libya, but it became extinct around the 1870s"
answer = predict_answer(context, question)
print(f"Answer: {answer}")

#%%
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

model_name = "savasy/bert-base-turkish-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

context = "Bolu, Türkiye'nin Karadeniz bölgesinde yer alan bir ildir."
question = "Bolu nerededir?"

inputs = tokenizer.encode_plus(question, context, return_tensors="pt")
outputs = model(**inputs)

answer_start = torch.argmax(outputs.start_logits)
answer_end = torch.argmax(outputs.end_logits) + 1

answer = tokenizer.convert_tokens_to_string(
    tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
)
print("Cevap:", answer)