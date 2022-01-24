#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 11:16:52 2022

@author: adernild
"""
#%% Importing libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import datetime
import PIL
import os
from sklearn.utils import class_weight 

#%% Finding average image size for resizing of images
files = os.listdir('data/DIDA_1')

y = []
x = []

for f in files:
    image = PIL.Image.open(os.path.join('data/DIDA_1/' + f))
    width, height = image.size
    y.append(height)
    x.append(width)

y_avg = int(np.sum(y) / len(y))
x_avg = int(np.sum(x) / len(x))

print(f'Smallest height: {np.array(y).min()}px, smallest width: {np.array(x).min()}px, largest height: {np.array(y).max()}px, largest width: {np.array(x).max()}px')
print(f'Average height: {y_avg}px, average width: {x_avg}px')

input_shape=(y_avg, x_avg, 3)

#%% Importing data
df = pd.read_csv('data/data.csv')

datagen = ImageDataGenerator(rescale=1/255.0)
datagen_test = ImageDataGenerator(rescale=1/255.0)
batch_size = 32

train_gen = datagen.flow_from_directory('data/split/CC/train', 
                                        batch_size=batch_size,
                                        shuffle=True,
                                        seed=1234,
                                        target_size=input_shape[0:2],
                                        class_mode='binary')

val_gen = datagen.flow_from_directory('data/split/CC/val', 
                                      batch_size=batch_size,
                                      shuffle=True,
                                      seed=1234,
                                      target_size=input_shape[0:2],
                                      class_mode='binary')

test_gen = datagen.flow_from_directory('data/split/CC/test')

#%%
modelCC = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=input_shape, name='conv_1'),
    Conv2D(32, (3,3), activation='relu', name='conv_2'),
    MaxPooling2D(2,2),
    Conv2D(32, (3,3), activation='relu', name='conv_3'),
    Conv2D(64, (3,3), activation='relu', name='conv_4'),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

modelCC.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=[tf.keras.metrics.AUC(name='auc')])

modelCC.summary()

#%% Callbacks
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=0)  # tensorboard --logdir ./logs
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

checkpoint_filepath = './checkpoints/checkpoint-modelCC'

modelCheckpoint = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_auc',
    mode='max',
    save_best_only=True) # Saving best checkpoint

earlyStop = EarlyStopping(
    monitor='val_loss',
    patience=15) # stopping training when val_loss doesn't decrease in 15 epochs

reduceLR = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=4) # reducing learning rate when val_loss doesn't improve for 3 epochs

#%% Training
# computing class weights
class_weights = class_weight.compute_class_weight(
           'balanced',
            np.unique(train_gen.classes), 
            train_gen.classes)

train_class_weights = dict(enumerate(class_weights))

STEP_SIZE_TRAIN=train_gen.n//train_gen.batch_size
STEP_SIZE_VALID=val_gen.n//val_gen.batch_size

modelCC.fit(train_gen,
            steps_per_epoch=STEP_SIZE_TRAIN,
            validation_data=val_gen,
            validation_steps=STEP_SIZE_VALID,
            epochs=100,
            callbacks=[tensorboard_callback, modelCheckpoint, earlyStop, reduceLR],
            class_weight=train_class_weights)
