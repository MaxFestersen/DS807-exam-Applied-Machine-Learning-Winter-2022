#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 11:01:17 2022

@author: adernild
"""

import os
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#%%
df = pd.read_csv('labels/Digit_String_2_labels_formatted.csv')
df['CC'] = df['CC'].astype(str)
df['D'] = df['D'].astype(str)
df['Y'] = df['Y'].astype(str)
datagen = ImageDataGenerator(rescale=1/255.0)

test_gen_CC = datagen.flow_from_dataframe(df,
                                       x_col='path',
                                       y_col='CC',
                                       class_mode='binary',
                                       batch_size=1,
                                       color_mode='grayscale',
                                       target_size=(32,62),
                                       shuffle=False)

test_gen_D = datagen.flow_from_dataframe(df,
                                       x_col='path',
                                       y_col='D',
                                       class_mode='categorical',
                                       batch_size=1,
                                       color_mode='grayscale',
                                       target_size=(32,62),
                                       shuffle=False)

test_gen_Y = datagen.flow_from_dataframe(df,
                                       x_col='path',
                                       y_col='Y',
                                       class_mode='categorical',
                                       batch_size=1,
                                       color_mode='grayscale',
                                       target_size=(32,62),
                                       shuffle=False)

#%%
modelCC = tf.keras.models.load_model('models/modelCC')
modelD = tf.keras.models.load_model('models/modelD_aug')
modelY = tf.keras.models.load_model('models/modelY_aug')
#%%

y_hat_CC = modelCC.predict(test_gen_CC)
y_hat_D = modelD.predict(test_gen_D)
y_hat_Y = modelY.predict(test_gen_Y)

#%%
if not os.path.exists('predictions/'):
    os.makedirs('predictions/')
np.save('predictions/y_hat_CC.npy', y_hat_CC)
np.save('predictions/y_hat_D.npy', y_hat_D)
np.save('predictions/y_hat_Y.npy', y_hat_Y)

