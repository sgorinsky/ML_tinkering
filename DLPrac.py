#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 20:59:16 2018

@author: samgorinsky
"""

import keras
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential

df = pd.read_csv('mnist.csv')

model = Sequential()

#adding 2 layers w/ relu activation + output layer to neural network
model.add(Dense(50, ))