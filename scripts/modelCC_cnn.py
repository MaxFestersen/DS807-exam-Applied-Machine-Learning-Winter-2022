#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 11:16:52 2022

@author: adernild
"""
#%% Importing libraries
import numpy as np
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import AUC
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

train_gen = datagen.flow_from_directory('data/split/CC/train', 
                                        batch_size=batch_size,
                                        shuffle=True,
                                        seed=1234,
                                        class_mode='binary',
                                        color_mode='grayscale')

val_gen = datagen.flow_from_directory('data/split/CC/val', 
                                      batch_size=batch_size,
                                      shuffle=True,
                                      seed=1234,
                                      class_mode='binary',
                                      color_mode='grayscale')

test_gen = datagen.flow_from_directory('data/split/CC/test',
                                       batch_size=batch_size,
                                       class_mode='binary',
                                       color_mode='grayscale')

input_shape=(32, 62, 1)

STEP_SIZE_TRAIN=train_gen.n//train_gen.batch_size
STEP_SIZE_VALID=val_gen.n//val_gen.batch_size

#%% Setting up hyperparameter tuning
HP_NUM_UNITS_1 = hp.HParam('num_units_1', hp.Discrete([8,16]))
HP_NUM_UNITS_2 = hp.HParam('num_units_2', hp.Discrete([16,32]))
HP_NUM_UNITS_3 = hp.HParam('num_units_3', hp.Discrete([128,256]))
HP_ACT_FUNC = hp.HParam('activation_func', hp.Discrete(['relu', 'tanh']))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

METRIC_PRC = tf.keras.metrics.AUC(name='prc', curve='PR')

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
    hp.hparams_config(
        hparams=[HP_NUM_UNITS_1, HP_NUM_UNITS_2, HP_NUM_UNITS_3, HP_ACT_FUNC, HP_OPTIMIZER],
        metrics=[hp.Metric(METRIC_PRC.name, display_name=METRIC_PRC.name)],
    )

def train_model(hparams):
    model = Sequential([
        Conv2D(hparams[HP_NUM_UNITS_1], (3,3), activation=hparams[HP_ACT_FUNC], input_shape=input_shape),
        MaxPooling2D(2,2),
        Conv2D(hparams[HP_NUM_UNITS_2], (3,3), activation=hparams[HP_ACT_FUNC]),
        Flatten(),
        Dense(hparams[HP_NUM_UNITS_3], activation=hparams[HP_ACT_FUNC]),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=hparams[HP_OPTIMIZER],
        loss='binary_crossentropy',
        metrics=[METRIC_PRC],
    )
    
    class_weights = class_weight.compute_class_weight(
               'balanced',
                np.unique(train_gen.classes), 
                train_gen.classes)

    train_class_weights = dict(enumerate(class_weights))
    
    model.fit(train_gen, 
              epochs=10,
              steps_per_epoch=STEP_SIZE_TRAIN,
              class_weight=train_class_weights)
    _, prc = model.evaluate()
    return prc

def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)
        prc = train_model(hparams)
        tf.summary.scalar(METRIC_PRC, prc, step=1)

#%% Running hyperparameter tuning

session_num = 0

for num_units_1 in HP_NUM_UNITS_1.domain.values:
    for num_units_2 in HP_NUM_UNITS_2.domain.values:
        for num_units_3 in HP_NUM_UNITS_3.domain.values:
            for activation in HP_ACT_FUNC.domain.values:
                for optimizer in HP_OPTIMIZER.domain.values:
                    hparams = {
                        HP_NUM_UNITS_1: num_units_1,
                        HP_NUM_UNITS_2: num_units_2,
                        HP_NUM_UNITS_3: num_units_3,
                        HP_ACT_FUNC: activation,
                        HP_OPTIMIZER: optimizer,
                    }
                    run_name = "run-%d" % session_num
                    print('--- Starting trail: %s' % run_name)
                    print({h.name: hparams[h] for h in hparams})
                    run('logs/hparam_tuning/' + run_name, hparams)
                    session_num += 1

#%% final model (not decided yet)
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

METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
      tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

modelCC.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=METRICS)

modelCC.summary()

#%% Callbacks
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=0)  # tensorboard --logdir ./logs
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

checkpoint_filepath = './checkpoints/checkpoint-modelCC'

modelCheckpoint = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_prc',
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

modelCC.fit(train_gen,
            steps_per_epoch=STEP_SIZE_TRAIN,
            validation_data=val_gen,
            validation_steps=STEP_SIZE_VALID,
            epochs=100,
            callbacks=[tensorboard_callback, modelCheckpoint, earlyStop, reduceLR],
            class_weight=train_class_weights)
