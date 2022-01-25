#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 16:42:42 2022

@author: adernild
"""

#%% Importing libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import datetime
from sklearn.utils import class_weight 
import os


# Set path to parrent location of current file
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
os.chdir("../")

#%% Importing data
datagen = ImageDataGenerator(rescale=1/255.0)
datagen_test = ImageDataGenerator(rescale=1/255.0)
batch_size = 32

train_gen = datagen.flow_from_directory('data/split/D/train', 
                                        batch_size=batch_size,
                                        shuffle=True,
                                        seed=1234,
                                        target_size=(130, 250),
                                        class_mode='categorical')

val_gen = datagen.flow_from_directory('data/split/D/val', 
                                      batch_size=batch_size,
                                      shuffle=True,
                                      seed=1234,
                                      target_size=(130, 250),
                                      class_mode='categorical')

test_gen = datagen.flow_from_directory('data/split/D/test')

input_shape=(130, 250, 3)

#%%
modelD = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=input_shape, name='conv_1'),
    Conv2D(32, (3,3), activation='relu', name='conv_2'),
    MaxPooling2D(2,2),
    Conv2D(32, (3,3), activation='relu', name='conv_3'),
    Conv2D(64, (3,3), activation='relu', name='conv_4'),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

modelD.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

modelD.summary()

#%% Callbacks
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=0)  # tensorboard --logdir ./logs
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

checkpoint_filepath = './checkpoints/checkpoint-modelD'

modelCheckpoint = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
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

modelD.fit(train_gen,
            steps_per_epoch=STEP_SIZE_TRAIN,
            validation_data=val_gen,
            validation_steps=STEP_SIZE_VALID,
            epochs=100,
            callbacks=[tensorboard_callback, modelCheckpoint, earlyStop, reduceLR],
            class_weight=train_class_weights)