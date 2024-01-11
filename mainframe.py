"""
Created on 
Wed Nov 23 23:39:42 2022
@author: parth
"""

from tkinter import *
import tkinter as tk
import tensorflow as tf
from tensorflow import keras
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import spacy
from nltk.corpus import stopwords

def find_features(document):
    features = []
    for i in document:
        doc = spmodel(i.lower())
        text = [j.lemma_ for j in doc if j.lemma_ not in stop_words]
        features.append(text)
        
    features = tokenizer.texts_to_sequences(features)
    features = [[j for j in i if j<=8000]for i in features]
    features = pad_sequences(features, maxlen=136)
            
    return features
        
        

def find_cat():
    x = t1.get(1.0, "end-1c")
    x = x.split("///")
    X = find_features(x)
   
    
    
    
    y_pred = model.predict(X, verbose=0)
    ans = np.argmax(y_pred, axis = 1)
    ans = labelencoder.classes_[ans]
    
    show = ""
    
    for i in range(len(ans)):
        show += str(i+1)+" "+ans[i]+"\n"
    
    var.set(show)
    
def clear():
    t1.delete(1.0, "end-1c")
    var.set("")
   
 
spmodel = spacy.load("en_core_web_sm")
model = keras.models.load_model("3-conv128-3-dense256-1680808526-weighted.h5")
stop_words = stopwords.words("english")

with open("tokenizer.pickle", "rb") as file: 
    tokenizer= pickle.load(file)
    
with open("labelencoder.pickle", "rb") as file:
    labelencoder = pickle.load(file)
    
window = Tk()
window.geometry("700x700")

var = StringVar()

lab2= Label(window , text="If want to find category of multiple news use /// to seperate them", font=("Comic Sans MS", 15, "bold"))
lab2.grid(column=1, row=0, padx=30, pady=20)
    
t1 = Text(window, height=15, width=65, font=("Comic Sans MS", 15))
t1.grid(column=1, row=3, padx=30)


button = Button(window, text="Find Category", command=lambda: find_cat(), font=("Comic Sans MS", 15, "bold"))
button.place(x=130, y=500)

button2 = Button(window, text="Clear", command= lambda: clear(), font=("Comic Sans MS", 15, "bold"))
button2.place(x=550, y=500)

res = Label(window, height=10, width=65, textvariable=var, borderwidth=2, relief="solid", font=("Comic Sans MS", 15, "bold"))
res.place(x=40, y=550)


window.mainloop()
