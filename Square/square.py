# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 21:22:08 2020

@author: Casper
"""

import numpy as np
import matplotlib.pyplot as plt

# %%
epoch=1200
np.random.seed(32432234)
x = np.random.random((10000,1))*100-50
y = x**2

print("x,y:\n", np.concatenate((x,y),axis=1))

plt.plot(x, y, 'o')
# %%

split=0.8
split_point=int(x.shape[0]*split) 
X_train=x[:split_point]
X_test=x[split_point:]
Y_train=y[:split_point]
Y_test=y[split_point:]
print(X_train.shape)

# %%
from keras.models import *
from keras.layers import *
from keras.regularizers import *
import tensorflow as tf

def baseline_model(X_train,Y_train):
    model=Sequential()
    model.add(Dense(8,activation="relu",kernel_regularizer=l2(0.001),input_shape=(1,)))
    model.add(Dense(8,activation="relu",kernel_regularizer=l2(0.001)))
    model.add(Dense(1))
    
    model.compile(optimizer="adam",loss="mse")
    return model

def baseline_model_norm(X_train,Y_train):
    model=Sequential()
    model.add(Dense(8,activation="relu",kernel_regularizer=l2(0.001),input_shape=(1,)))
    model.add(Dense(8,activation="relu",kernel_regularizer=l2(0.001)))
    model.add(Dense(1,activation="sigmoid"))
    model.compike(optimizer="adam",loss="mse")
    return model

# %%
model = baseline_model(X_train, Y_train)
model.summary()


hist = model.fit(X_train, Y_train,validation_split=0.2,
             epochs= 1200,
             batch_size=256)
print("train end")

# %%
plt.title('Loss / Mean Squared Error')
plt.plot(hist.history['loss'], label='train')
plt.plot(hist.history['val_loss'], label='validation')
plt.legend()
plt.show()

# %%

