# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 01:37:23 2022

@author: Anders
"""
#%% Importing libraries
import numpy as np
import tensorflow as tf
from collections import Counter
import pandas as pd
from numpy import load
import seaborn
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
datagen = ImageDataGenerator() # EfficientNet input range [0,255]
datagen_test = ImageDataGenerator() # EfficientNet input range [0,255]
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
for i in ['CC','D','Y']:
    # ImageDataGenerator shifting
    datagen = ImageDataGenerator()
    # iterator
    aug_iter = datagen.flow_from_directory(f'data/split/{i}/test', 
                                           batch_size=1,
                                           shuffle=True,
                                           seed=1234,
                                           class_mode='binary',
                                           color_mode='grayscale',
                                           target_size=(32, 62))
    
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(15,15))
    for i in range(4):
         image = next(aug_iter)[0].astype('uint8')
        
         image = np.squeeze(image)
        
         ax[i].imshow(image, cmap='Blues')
         ax[i].axis('off')
#%%
def plot_confusion_matrix(x):
    #x = df_confusion.reindex(columns=[x for x in range(len(Counter(y_test)))], fill_value=0) # makes sure 
    seaborn.heatmap(x, annot=True, fmt='d', cmap='Blues')
#%%
def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()
#%%    
def plot_loss(hist):
    plt.plot(hist.history["loss"])
    plt.plot(hist.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()
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
def build_model():
    inputs = tf.keras.layers.Input(shape=(32, 62, 3))
    x = data_augmentation(inputs)
    model = tf.keras.applications.efficientnet.EfficientNetB0(
        include_top=False, 
        input_tensor=x, 
        weights="imagenet")
    
    model.trainable = False

    # Rebuild top
    x = model.output
    x = tf.keras.layers.Flatten(input_shape=(1, 1, 2560))(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    # Compile
    model = tf.keras.Model(inputs=model.input, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss='binary_crossentropy',
        metrics=['accuracy',METRIC_KAPPA]
    )
    return model

#%% train head for CC
model = build_model()

epochs = 20  # @param {type: "slider", min:8, max:80}
hist = model.fit(train_gen_CC, validation_data=val_gen_CC, epochs=epochs, verbose=1)
plot_hist(hist)

#%%
KERAS_MODEL_NAME = "FWL_Head.hdf5"
#model.save('exercise-nn-saveload')
tf.keras.models.save_model(model, KERAS_MODEL_NAME)
model_reload = tf.keras.models.load_model('FWL_Head.hdf5')
#%%
def unfreeze_model(model):
    # We unfreeze layers while leaving BatchNorm layers frozen
    for layer in model.layers[:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy',METRIC_KAPPA]
    )

#%%
unfreeze_model(model)

epochs = 10  
hist = model.fit(train_gen_CC, validation_data=val_gen_CC, epochs=epochs, verbose=1)
plot_hist(hist)
plot_loss(hist)
KERAS_MODEL_NAME = "FWL_model_final.hdf5"
#model.save('exercise-nn-saveload')
tf.keras.models.save_model(model, KERAS_MODEL_NAME)
model_reload = tf.keras.models.load_model('FWL_model_final.hdf5')
#%%
model_reload.evaluate(test_gen_CC)
#%%
y_test_hat = np.where(model_reload.predict(test_gen_CC) > 0.5, 1, 0).flatten()
df_confusion = pd.crosstab(test_gen_CC.classes, y_test_hat, rownames=['Actual'], colnames=['Predicted'],dropna=False)
plot_confusion_matrix(df_confusion)




