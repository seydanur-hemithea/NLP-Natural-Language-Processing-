# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 20:42:22 2025

@author: asus
"""

#Solve Classification problem(setiment analysis in NLP)with RNN
#restaurant comments 

import numpy as np 
import pandas as pd
from gensim.models import Word2Vec
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import SimpleRNN,Dense,Embedding
from keras_preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


data = (
    {
        "text": [
            "Yemekler çok lezzetliydi",
            "Servis çok yavaştı",
            "Garsonlar çok ilgiliydi",
            "Tatlılar bayattı",
            "Mekan çok şık ve temizdi",
            "Yemekler soğuktu",
            "Sunum çok estetikti",
            "Siparişimiz karıştı",
            "Tatlılar harikaydı",
            "Garson kaba davrandı",
            "Mekan atmosferi çok huzurluydu",
            "Yemekler çok yağlıydı",
            "Çalışanlar güler yüzlüydü",
            "Hesap yanlış geldi",
            "Lezzet muhteşemdi, tekrar geleceğim",
            "Sandalyeler rahatsızdı",
            "İçecekler çok ferahlatıcıydı",
            "Yemekler tuzsuzdu",
            "Mekan çok ferah ve aydınlıktı",
            "Garson ilgisizdi",
            "Tatlılar tam kararındaydı",
            "Masa çok küçüktü",
            "Yemekler özenle hazırlanmıştı",
            "Servis personeli yetersizdi",
            "Sunum çok yaratıcıydı",
            "Yemekler yanmıştı",
            "Mekan dekorasyonu çok etkileyiciydi",
            "Garson siparişi yanlış getirdi",
            "Tatlılar çok hafifti",
            "Yemekler çok baharatlıydı",
            "Mekan çok sessiz ve huzurluydu",
            "Servis çok aceleciydi",
            "Yemekler tam zamanında geldi",
            "Garsonlar ilgisizdi",
            "Tatlılar çok şekerliydi",
            "Mekan çok temizdi",
            "Yemekler çok sıradandı",
            "Garsonlar çok yardımseverdi",
            "Yemekler çok soğuktu",
            "Tatlılar çok taze değildi",
            "Mekan çok kalabalıktı",
            "Garsonlar çok profesyoneldi",
            "Yemekler çok iyi pişmişti",
            "Servis çok hızlıydı",
            "Tatlılar çok hafifti",
            "Yemekler çok tuzluydu",
            "Mekan çok karışıktı",
            "Garsonlar çok nazikti",
            "Yemekler çok güzel kokuyordu",
            "Servis çok geç geldi"
        ],
        "label": [
            "positive","negative","positive","negative","positive","negative","positive","negative","positive","negative",
            "positive","negative","positive","negative","positive","negative","positive","negative","positive","negative",
            "positive","negative","positive","negative","positive","negative","positive","negative","positive","negative",
            "positive","negative","positive","negative","positive","negative","positive","negative","positive","negative",
            "positive","negative","positive","negative","positive","negative","positive","negative","positive","negative"
        ]
    }
)
df=pd.DataFrame(data)

tokenizer=Tokenizer()
tokenizer.fit_on_texts(df["text"])
sequences=tokenizer.texts_to_sequences(df["text"])
word_index=tokenizer.word_index
#padding process
maxlen=max(len(seq)for seq in sequences)
X=pad_sequences(sequences,maxlen=maxlen)
print(X.shape)

label_encoder=LabelEncoder()
y=label_encoder.fit_transform(df["label"])

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


sentences=[text.split() for text in df["text"]]
word2vec_model=Word2Vec(sentences,vector_size=50,window=5,min_count=1)

embedding_dim=50
embedding_matrix=np.zeros((len(word_index)+1,embedding_dim))
for word,i in word_index.items():
    if word in word2vec_model.wv:
        embedding_matrix[i]=word2vec_model.wv[word]

model=Sequential()
#embedding
model.add(Embedding(input_dim=len(word_index)+1,output_dim=embedding_dim,
                    weights=[embedding_matrix],
                    input_length=maxlen,
                    trainable=False
                    ))



#RNN
model.add(SimpleRNN(50,return_sequences=False))

#output layer
model.add(Dense(1,activation="sigmoid"))
#compile model
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
#train model
model.fit(X_train,y_train,epochs=13,batch_size=2,validation_data=(X_test,y_test))


#evaluate RNN
test_loss,test_accuracy=model.evaluate(X_test,y_test)
print(f"test loss:{test_loss}")
print(f"test accuracy:{test_accuracy}")

def classify_sentence(sentence):
    seq=tokenizer.texts_to_sequences([sentence])
    padded_seq=pad_sequences(seq,maxlen=maxlen)
    
    prediction=model.predict(padded_seq)
    
    predicted_class=(prediction>0.5).astype(int)
    label="positive" if predicted_class[0][0]==1 else "negative"
    
    return label

sentence="Garsonalrın tırnakları temizdi ve garsonlar kibardı ,yemklerde sıcak ve lezzetliydi"


result=classify_sentence(sentence)
print(f"result:{result}")
#%% LSTM
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

texts = [
    "Bugün hava çok güzel, dışarıda yürüyüş yapmayı düşünüyorum.",
    "Kitap okumak beni gerçekten mutlu ediyor.",
    "Sabah kahvemi içerken sessizliği dinlemeyi seviyorum.",
    "Müzik dinlemek ruhumu dinlendiriyor, özellikle yağmurlu günlerde.",
    "Kedim kucağıma geldiğinde tüm yorgunluğum geçiyor.",
    "Yemek yaparken zamanın nasıl geçtiğini anlamıyorum.",
    "Deniz kenarında yürümek bana huzur veriyor.",
    "Film izlerken karakterlerle duygusal bağ kuruyorum.",
    "Gün batımını izlemek içimde bir dinginlik yaratıyor.",
    "Bazen sadece sessizce oturmak bile iyi geliyor.",
    "Sabahları erken kalkmak zihnimi tazeliyor.",
    "Bir fincan çay, günün en güzel başlangıcı oluyor.",
    "Yağmur sesi beni çocukluğuma götürüyor.",
    "Kalabalıktan uzak olmak bazen en büyük lüks.",
    "Yıldızları izlemek geceyi anlamlı kılıyor.",
    "Bir dostla yapılan sohbet, ruhu besliyor.",
    "Sessiz bir kütüphane, düşüncelerime ev oluyor.",
    "Yürürken düşünmek, zihnimi toparlıyor.",
    "Bir gülümseme bazen tüm günü güzelleştiriyor.",
    "Kahve kokusu beni her zaman rahatlatıyor.",
    "Pencere kenarında oturmak bana iyi geliyor.",
    "Bir günlüğe yazmak, içimi dökmenin en güzel yolu.",
    "Küçük şeyler bazen en büyük mutluluğu getiriyor.",
    "Bir şarkı, geçmişi yeniden yaşatabiliyor.",
    "Evde yalnız kalmak bana huzur veriyor.",
    "Bir resme bakmak, hayal gücümü harekete geçiriyor.",
    "Sıcak bir battaniye, soğuk günlerin kurtarıcısı.",
    "Bir çiçeğin açışını izlemek sabrı öğretiyor.",
    "Kendi kendime konuşmak bazen en iyi terapi.",
    "Bir mektup yazmak, duyguları kelimelere dökmek demek.",
    "Karanlıkta yürümek, düşüncelerimi netleştiriyor.",
    "Bir mumun ışığı, geceye anlam katıyor.",
    "Küçük bir hediye, büyük bir tebessüm yaratıyor.",
    "Bir fotoğraf, geçmişin sessiz tanığı oluyor.",
    "Sakin bir akşam, günün yorgunluğunu alıyor.",
    "Bir hayal kurmak, geleceğe umut veriyor.",
    "Bir kitapta kaybolmak, gerçeklerden kaçmak gibi.",
    "Bir tatlı yemek, ruhumu şımartıyor.",
    "Bir gün boyunca sessiz kalmak, içsel bir yolculuk gibi.",
    "Bir sokak lambası, geceye rehberlik ediyor.",
    "Bir çocuğun kahkahası, dünyayı güzelleştiriyor.",
    "Bir pencereyi açmak, yeni bir nefes gibi.",
    "Bir defterin boş sayfası, yeni bir başlangıç demek.",
    "Bir şairin dizeleri, kalbime dokunuyor.",
    "Bir rüzgar esintisi, geçmişi hatırlatıyor.",
    "Bir dostun sesi, yalnızlığı unutturuyor.",
    "Bir fincan salep, kışa sıcaklık katıyor.",
    "Bir yürüyüş, zihnimi temizliyor.",
    "Bir gün batımı, hayatın geçiciliğini hatırlatıyor.",
    "Bir kuşun ötüşü, sabahı selamlıyor.",
    "Bir sokak kedisi, günümü güzelleştiriyor.",
    "Bir anı, kalbimde iz bırakıyor.",
    "Bir şiir, duygularımı tercüme ediyor.",
    "Bir bulutun şekli, hayal gücümü besliyor.",
    "Bir sessizlik, bazen en güçlü cevaptır.",
    "Bir gözyaşı, içimdeki yükü hafifletiyor.",
    "Bir tebessüm, kalpleri yakınlaştırıyor.",
    "Bir melodi, ruhumu sarıyor.",
    "Bir gün, sadece kendime ait olsun istiyorum.",
    "Bir sabah, umutla uyanmak istiyorum.",
    "Bir gece, yıldızlarla konuşmak istiyorum.",
    "Bir an, sadece durmak ve hissetmek istiyorum.",
    "Bir kitap, beni başka dünyalara götürüyor.",
    "Bir kahve, düşüncelerimi berraklaştırıyor.",
    "Bir yürüyüş, içimdeki karmaşayı dağıtıyor.",
    "Bir dost, en karanlık anımda ışık oluyor.",
    "Bir anı, beni gülümsetiyor.",
    "Bir kelime, bazen her şeyi değiştiriyor.",
    "Bir bakış, kalbime dokunuyor.",
    "Bir gün, sadece sessizliği dinlemek istiyorum.",
    "Bir gece, sadece hayal kurmak istiyorum.",
    "Bir sabah, sadece huzurla uyanmak istiyorum.",
    "Bir an, sadece kendimi hissetmek istiyorum.",
    "Bir şarkı, beni geçmişe götürüyor.",
    "Bir resim, duygularımı anlatıyor.",
    "Bir şiir, içimi döküyor.",
    "Bir yürüyüş, beni kendime getiriyor.",
    "Bir kahve, beni sakinleştiriyor.",
    "Bir kitap, bana yeni bakış açıları kazandırıyor.",
    "Bir gün, sadece kendimle olmak istiyorum.",
    "Bir gece, sadece düşünmek istiyorum.",
    "Bir sabah, sadece gülümsemek istiyorum.",
    "Bir an, sadece var olmak istiyorum.",
    "Bir dost, bana güç veriyor.",
    "Bir kelime, beni iyileştiriyor.",
    "Bir bakış, beni anlıyor.",
    "Bir gün, sadece sevmek istiyorum.",
    "Bir gece, sadece sarılmak istiyorum.",
    "Bir sabah, sadece umutla başlamak istiyorum.",
    "Bir an, sadece nefes almak istiyorum.",
    "Bir şarkı, beni sarıyor.",
    "Bir resim, beni anlatıyor.",
    "Bir şiir, beni tamamlıyor.",
    "Bir yürüyüş, beni özgürleştiriyor.",
    "Bir kahve, beni dinlendiriyor.",
    "Bir kitap, beni büyütüyor.",
    "Bir gün, sadece hayal etmek istiyorum.",
    "Bir gece, sadece yıldızlara bakmak istiyorum.",
    "Bir sabah, sadece sessizliği duymak istiyorum.",
    "Bir an, sadece kendimi bulmak istiyorum.",
    "Bir dost, bana huzur veriyor.",
    "Bir kelime, bana umut veriyor.",
    "Bir bakış, bana sevgi veriyor."
]


tokenizer=Tokenizer()
tokenizer.fit_on_texts(texts)#word freq
total_words=len(tokenizer.word_index)+1#total words


# create n-gram arrays and apply padding
input_sequences=[]
for text in texts:
    #Convert texts to word indices
    token_list=tokenizer.texts_to_sequences([text])[0]
    
    
    #create n gram array per text
    for i in range(1,len(token_list)):
        n_gram_sequence=token_list[:i+1]
        input_sequences.append(n_gram_sequence)
max_sequence_length=max(len(x) for x in input_sequences)
#must same length 
input_sequences=pad_sequences(input_sequences,maxlen=max_sequence_length,padding="pre")

X=input_sequences[:,:-1]
y=input_sequences[:,-1]

y=tf.keras.utils.to_categorical(y,num_classes=total_words)#one hot encoding
#LSTM model

model=Sequential()
model.add(Embedding(total_words,50,input_length=X.shape[1]))
#lstm
model.add(LSTM(100,return_sequences=False))
#output
model.add(Dense(total_words,activation="softmax"))

#compile
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])#model training
model.fit(X,y,epochs=50,verbose=1)

#model prediction
def generate_text(seed_text,next_words):
    for _ in range(next_words):
        #input text vectorizing
        token_list=tokenizer.texts_to_sequences([seed_text])[0]
        #padding
        token_list=pad_sequences([token_list],maxlen=max_sequence_length-1,padding="pre")
        #prediction
        
        predicted_probabilities=model.predict(token_list,verbose=0)
        #words indexes which  has highest probability finding
        predicted_word_index=np.argmax(predicted_probabilities,axis=-1)
        #real word is finded with tokenizer from word_index
        predicted_word=tokenizer.index_word[predicted_word_index[0]]
        #predictedw word is added seed_text
        seed_text=seed_text+" "+predicted_word
        
    return seed_text

seed_text="bugün yağmur " 
print(generate_text(seed_text,5))       
    
#%%gpt2 -LLaMA
from transformers import GPT2LMHeadModel,GPT2Tokenizer
model_name="gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)



text="I go to school for"
#tokenization
inputs=tokenizer.encode(text,return_tensors="pt")
outputs=model.generate(inputs,max_length=55)#inputs=start point model,max_length=max token count

#tokens must be readable

generated_text=tokenizer.decode(outputs[0],skip_special_tokens=True)#special tokens out(start and end tokens)
print(generated_text)

#llama

from transformers import AutoTokenizer,AutoModelForCausalLM#llama
model_name="huggyllama/llama-7b"#!!13 gigabyte 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)



text="I go to school for"
#tokenization
inputs=tokenizer(text,return_tensors="pt")
outputs=model.generate(inputs.input_ids,max_length=55)

#tokens must be readable

generated_text=tokenizer.decode(outputs[0],skip_special_tokens=True)#special tokens out(start and end tokens)
print(generated_text)




























