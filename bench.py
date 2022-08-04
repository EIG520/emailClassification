import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

model = keras.models.load_model('model.h5')

# Hyperparameters
num_words = 1000
oov_token = '<UNK>'
pad_type = 'post'
trunc_type = 'post'

hamArr = []
i = 0
for ham in os.listdir("Dataset/20021010_easy_ham"):
    i+=1
    if(i>500):break
    with open(f"Dataset/20021010_easy_ham/{ham}",mode = "rb") as tham:
        toAdd = tham.read().decode(errors='ignore')
        hamArr += [toAdd] 

spamArr = []
i=0
for spam in os.listdir("Dataset/20021010_spam"):
    i+=1
    if(i>500):break
    with open(f"Dataset/20021010_spam/{spam}",mode="rb") as tspam:
        toAdd = tspam.read().decode(errors='ignore')
        spamArr += [toAdd]
print(len(hamArr))
print(len(spamArr))
y_train = []
badx_train = []
for i in range(0,500):
    badx_train.append(hamArr[i])
    badx_train.append(spamArr[i])
    y_train.append(0)
    y_train.append(1)

# Tokenizing
tokenizer = Tokenizer(num_words = num_words, oov_token = oov_token)
tokenizer.fit_on_texts(badx_train)

word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(badx_train[300:400])
train_sequences2 = tokenizer.texts_to_sequences(badx_train)
y_train = y_train[300:400]

maxlen = max([len(x) for x in train_sequences2])

train_padded = pad_sequences(train_sequences, padding=pad_type, truncating=trunc_type, maxlen=maxlen)

train_padded = np.asarray(train_padded)
y_train = np.asarray(y_train)

val_loss,val_acc = model.evaluate(train_padded,y_train)