#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 17:05:58 2018

@author: samgorinsky
"""

import keras
from keras.layers import Dense
from keras.models import Sequential
import pandas as pd
import numpy as np

df = pd.read_csv('mnist.txt')

predictors = df.drop(['5'],1)
target = df['5']

n_cols = predictors.shape[1]

# Set up the model: model
model = Sequential()

# Add the first hidden layer
model.add(Dense(50, activation = 'relu', input_shape = (784,)))

# Add the second hidden layer
model.add(Dense(50, activation = 'relu'))

# Add the output layer
model.add(Dense(10, activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(predictors, target, validation_split = .3)