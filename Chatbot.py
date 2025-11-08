# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 16:41:17 2025

@author: asus
"""

import  openai

openai.api_key="your_api_key"#https://platform.openai/api-keys

def chat_with_gpt(prompt,history_list):
    responce=openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":f"This is my message:{prompt}.History of message:{history_list}"}]
                  )
    return responce.choices[0].message.content.strip()
if __name__=="__main__":
    history_list=[]
    while True:
         user_input=input("User's messages:")
         if user_input.lower() in ["exit","q"]:
             print("Speaking is completed")
             break
         history_list.append(user_input)
         responce=chat_with_gpt(user_input,history_list)
         print("Chatbot:{responce}")
         
             
    
