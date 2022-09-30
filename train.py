
# импортируем
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GRU, LSTM
from tensorflow.keras import utils
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import pandas as pd
import numpy as np
import json


# ограничиваем колличество слов в словаре
num_words = 11000

# задаем длину вектора кодирования предложения
max_review_len = 21

# загрузаем данные
train = pd.read_csv('./train_issues.csv', )

# перемешиваем данные
train = train.sample(frac=1)
train.reset_index(drop=True, inplace=True)


# очищаем данные от спецсимволов и приводим к нижнему регистру
import re
reg = re.compile('[^a-zA-Zа-яА-Я0-9 ]')

x_train = []
for string in train["summary"]:
    string = ' '.join(reg.sub(' ', string).split())
    x_tr = []
    a = 0
    for x_w in string.lower().split(' '):
        x_tr.append(x_w)
        if a >= (max_review_len-1):
            break
        a = a + 1
    x_train.append( ' '.join( x_tr ) )
x_train[:5]


# ограничиваем максимальную длинну выходного параметра так, как минимальные и максимальные значения отличаются на порядки
# и переводим значения из абсолютных едениц в относительные в диапазоне от 0 до 1. Так нейронная сеть будет гораздо лучше обучаться.
min_time = 0
max_time = 80000
y_train_y = []
n = 0
for y_t in train["overall_worklogs"]:
    if y_t < min_time:
        y_train_y.append(min_time/max_time)
    else:
        if y_t > max_time:
            y_train_y.append(max_time/max_time)
        else:
            y_train_y.append(y_t/max_time)
        
y_train = pd.DataFrame({ "overall_worklogs" : y_train_y })


# создаем словарь
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts( x_train )


# преобразуем предложения в числовые векторы
sequences = tokenizer.texts_to_sequences( x_train )

# дополняем векторы нулями так, чтобы все векторы имели одинаковую длину
x_train = pad_sequences(sequences, maxlen=max_review_len)


# создаем свою собственную метрику
from tensorflow.keras import backend as K
def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

# создаем свою собственную функцию потерь
def my_loss_fn(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 

    return ( SS_res/(SS_tot + K.epsilon()) )


# создаем свою собственную функцию активации, которая никогда не будет выдавать отрицательные значения
def custom_activation(x):
    return K.sqrt(K.square(x))

# создаем модель на базе рекурентного слоя LSTM так, как LSTM хорошо работает с последовательностями.
model = Sequential()
model.add(Embedding(num_words, 2, input_length=max_review_len))
model.add(LSTM(128))
model.add(Dense(1, activation=custom_activation))

learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate)
model.compile(
    optimizer=optimizer, 
    loss=my_loss_fn, 
    metrics=[ coeff_determination ]
  )




# обучаем нейронную сеть переодически резко повышая learning_rate чтобы встряхнуть нейросеть, чтоб она 
# пыталась выйти из локального минимума и пыталась найти более глобальный градиент
# если не выходит из локального минимума, то начинаем обучать заново
# ближе к концу уменьшаем learning_rate, чтоб точнее поймать момент перед переобучением.
checkpoint_callback = ModelCheckpoint(
    filepath="{val_coeff_determination:.4f}-{epoch:03d}-{coeff_determination:.4f}.h5",
    monitor='val_coeff_determination',
    mode='max',
    save_best_only=True,
    verbose=1
)

history = model.fit(x_train, 
                    y_train, 
                    epochs=350,
                    batch_size=4096,
                    validation_split=0.86,
                    callbacks=[checkpoint_callback])