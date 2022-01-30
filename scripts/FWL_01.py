# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 13:08:57 2022

@author: A
"""
#%% Importing libraries
import numpy as np
import tensorflow as tf
import pandas as pd
import tensorflow_addons as tfa
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import datetime
from sklearn.utils import class_weight 
import matplotlib.pyplot as plt
import os
import seaborn as sns

# Set path to parrent location of current file
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
os.chdir("../")
#%% Importing data
#%%
%load_ext tensorboard
%tensorboard --logdir logs_exercise_nn_saveload --port 5678
#%%
datagen = ImageDataGenerator(samplewise_center=True)
datagen_test = ImageDataGenerator(samplewise_center=True)
batch_size = 64

train_gen_CC = datagen.flow_from_directory('data/split/CC/train', 
                                        batch_size=batch_size,
                                        shuffle=True,
                                        seed=1234,
                                        class_mode='binary',
                                        color_mode='grayscale',
                                        target_size=(32, 62))

val_gen_CC = datagen.flow_from_directory('data/split/CC/val', 
                                      batch_size=batch_size,
                                      shuffle=True,
                                      seed=1234,
                                      class_mode='binary',
                                      color_mode='grayscale',
                                      target_size=(32, 62))

test_gen_CC = datagen.flow_from_directory('data/split/CC/test',
                                       batch_size=batch_size,
                                       class_mode='binary',
                                       color_mode='grayscale',
                                       target_size=(32, 62),
                                       shuffle=False)

#%%
data_augmentation = Sequential(
    [
        tf.keras.layers.experimental.preprocessing.RandomRotation(factor=0.15, input_shape=(32, 62, 3)),
        tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
        tf.keras.layers.experimental.preprocessing.RandomFlip(),
        tf.keras.layers.experimental.preprocessing.RandomContrast(factor=0.1),
    ],
)
METRIC_KAPPA = tfa.metrics.CohenKappa(num_classes=2, name='kappa')
#%%
base_model = tf.keras.applications.MobileNetV2(
    #input_shape=(32,62,3),
    include_top=False, # cut off the head
    weights='imagenet', # pretrained on the ImageNet data
)
base_model.trainable = False # freeze the base model to not train it 
                             # not needed for pure feature extraction,
                             # but needed for the feature extraction 
                             # with multiple passes (as it is implemented here)

# Create new features
z_train = base_model.predict(2*train_gen_CC-1) # Rescaling
z_test = base_model.predict(2*test_gen_CC-1) # Rescaling
#%%
model_pfe = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(1, 1, 1280)), # flatten before fully connected part
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
model_pfe.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy',METRIC_KAPPA],
    )
model_pfe.summary()

#%%
history_pfe = model_pfe.fit(z_train, train_labels, validation_data=(z_test, test_labels), 
                            epochs=15, verbose=1)