import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


# Hyperparameters
num_words = 1000
oov_token = '<UNK>'
pad_type = 'post'
trunc_type = 'post'

# Download vocabulary data
vocab_file = keras.utils.get_file(
    origin="https://storage.googleapis.com/tensorflow/keras-nlp/examples/bert/bert_vocab_uncased.txt",
)

# Get data
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

train_sequences = tokenizer.texts_to_sequences(badx_train[0:300])
train_sequences2 = tokenizer.texts_to_sequences(badx_train)
y_train = y_train[0:300]

maxlen = max([len(x) for x in train_sequences2])

train_padded = pad_sequences(train_sequences, padding=pad_type, truncating=trunc_type, maxlen=maxlen)

train_padded = np.asarray(train_padded)
y_train = np.asarray(y_train)

# Actual Model

model = tf.keras.models.Sequential()

#input
model.add(tf.keras.layers.Flatten())
#hidden
model.add(tf.keras.layers.Dense(128,activation = 'relu'))
model.add(tf.keras.layers.Dense(128,activation = 'relu'))
model.add(tf.keras.layers.Dense(128,activation = 'relu'))
model.add(tf.keras.layers.Dense(128,activation = 'relu'))
#output
model.add(tf.keras.layers.Dense(2,activation=tf.nn.softmax))

#compiling/learning
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics =['accuracy'])

model.fit(train_padded,y_train,epochs = 30)

val_loss,val_acc = model.evaluate(train_padded,y_train)

model.save("model.h5")

with open("maxlen.txt","w") as maxlen:
    maxlen.write(str(maxlen))

input("The model is trained!\n press enter to exit")