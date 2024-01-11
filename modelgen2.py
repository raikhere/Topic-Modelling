import nltk
import pandas as pd
import pickle
import time
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from tensorflow import keras
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split, cross_val_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder


# model building imports
from keras.layers import Embedding, Flatten, Dense, Dropout
from keras.layers import Conv1D, SimpleRNN, Bidirectional, MaxPooling1D, GlobalMaxPool1D, LSTM, GRU, Input
from keras.models import Sequential
from keras.regularizers import L1L2
#from wordcloud import WordCloud, STOPWORDS



df = pd.read_json("newsdataset.json", lines=True)

with open("tokenizer.pickle", "rb") as file: 
    tokenizer= pickle.load(file)
    
with open("labelencoder.pickle", "rb") as file:
    labelencoder = pickle.load(file)
    
with open("onehotencoder.pickle", "rb") as file: 
    encoder = pickle.load(file)
    
    
X = df["reducedtoken"]
train_seq = tokenizer.texts_to_sequences(X)
train_padseq = pad_sequences(train_seq, maxlen=136)
Y = encoder.transform(np.array(df["category"]).reshape(-1,1)).toarray()

padseq_train, padseq_test, y_train, y_test = train_test_split(train_padseq, Y, test_size=0.2)

weight_class = {0: 1.8082155477031803,
 1: 1.0711145996860283,
 2: 1.5821449275362318,
 3: 2.404581497797357,
 4: 2.1112944816915937,
 5: 2.3898423817863397,
 6: 3.8117318435754193,
 7: 0.5098766969734712,
 8: 1.3150658528750403,
 9: 1.223125186734389,
 10: 1.9517520858164483,
 11: 0.9435980177480697,
 12: 2.070189633375474,
 13: 0.25008705213964993,
 14: 1.9219718309859155,
 15: 1.6764127764127765,
 16: 0.6878602033100899,
 17: 1.3114848630466123,
 18: 0.8281177303529889,
 19: 0.45928086610198016}

convlayers = []
numfilters = [256]
denselayers = [2, 3]
densesizes = [256, 512]

#2 256conv 3 256dense
#2 128conv 3 256dense
#2 256conv 3 512dense

#specs = [[3,256,3,256],[3,128,3,256],[3,256,3,512]]

specs = [[3, 128, 3, 256], [3, 256,3, 256]]


for spec in specs:
	log_dir = f"logs/{spec[0]}-conv{spec[1]}-{spec[2]}-dense{spec[3]}-{int(time.time())}-weightedBatch"
	checkpoint = keras.callbacks.ModelCheckpoint(f"{spec[0]}-conv{spec[1]}-{spec[2]}-dense{spec[3]}-{int(time.time())}-weightedBatch.h5", save_best_only=True)
	model = keras.Sequential()
	model.add(keras.layers.Input(shape=136))
	model.add(keras.layers.Embedding(8000, 200))
                
	for i in range(spec[0]):
		model.add(keras.layers.Conv1D(spec[1], 3, activation="relu"))
                
	model.add(keras.layers.Flatten())
	model.add(keras.layers.BatchNormalization())
                
	for i in range(spec[2]):
		model.add(keras.layers.Dense(spec[3], activation="LeakyReLU"))
		if i%2==0:
			model.add(keras.layers.ActivityRegularization(l2=0.03))
                        
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.Dense(20, activation="softmax"))
	model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
                
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
                
	model.fit(padseq_train, y_train, epochs=8, validation_split=0.2, callbacks=[tensorboard_callback,checkpoint], batch_size=20, class_weight=weight_class)

                
