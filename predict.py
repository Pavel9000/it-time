

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

# загружаем данные
test = pd.read_csv('./test_issues.csv', )

# очищаем данные от спецсимволов и приводим к нижнему регистру
import re
reg = re.compile('[^a-zA-Zа-яА-Я0-9 ]')

x_test = []
for string in test["summary"]:
    string = ' '.join(reg.sub(' ', string).split())
    x_tr = []
    a = 0
    for x_w in string.lower().split(' '):
        x_tr.append(x_w)
        if a >= (max_review_len-1):
            break
        a = a + 1
    x_test.append( ' '.join( x_tr ) )
x_test[:5]


# ограничиваем колличество слов в словаре
num_words = 11000

# задаем длину вектора кодирования предложения
max_review_len = 21

# загрузаем данные
train = pd.read_csv('./train_issues.csv', )

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

# создаем словарь
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts( x_train )


# преобразуем предложения в числовые векторы
sequence = tokenizer.texts_to_sequences( x_test )

# дополняем векторы нулями так, чтобы все векторы имели одинаковую длину
np.array( sequence )
data = pad_sequences(sequence, maxlen=max_review_len)



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


# загружаем модель
model = tf.keras.models.load_model('./model.h5', custom_objects={'my_loss_fn': my_loss_fn, 'coeff_determination': coeff_determination, 'custom_activation' : custom_activation})

# делаем предсказания
result = model.predict(data)

# переводим из относительных едениц в абсолютные
res = []
for rrr in result:
    res.append(rrr[0]*max_time)

# записываем результат в файл
pd.DataFrame({ "id" : test['id'],"overall_worklogs" : res }).to_csv('sample_solution_x.csv',index=False)