#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 16:42:42 2022

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
                                        class_mode='categorical',
                                        color_mode='grayscale',
                                        target_size=(32, 62))

val_gen = datagen.flow_from_directory('data/split/D/val', 
                                      batch_size=batch_size,
                                      shuffle=True,
                                      seed=1234,
                                      class_mode='categorical',
                                      color_mode='grayscale',
                                      target_size=(32, 62))

test_gen = datagen.flow_from_directory('data/split/D/test')

input_shape=(32, 62, 1)

#%% Setting up hyperparameter tuning
HP_NUM_UNITS_1 = hp.HParam('num_units_1', hp.Discrete([8,16,32]))
HP_NUM_UNITS_2 = hp.HParam('num_units_2', hp.Discrete([16,32,64]))
HP_NUM_UNITS_3 = hp.HParam('num_units_3', hp.Discrete([128,256,512]))
HP_ACT_FUNC = hp.HParam('activation_func', hp.Discrete(['relu', 'tanh']))
HP_ACT_FUNC_2 = hp.HParam('activation_func_2', hp.Discrete(['relu', 'tanh']))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

METRIC_PRC = tf.keras.metrics.AUC(name='prc', curve='PR')
METRIC_ACC = 'accuracy'

with tf.summary.create_file_writer('logs/hparam_tuning_D').as_default():
    hp.hparams_config(
        hparams=[HP_NUM_UNITS_1, HP_NUM_UNITS_2, HP_NUM_UNITS_3, HP_ACT_FUNC, HP_ACT_FUNC_2, HP_OPTIMIZER],
        metrics=[hp.Metric(METRIC_PRC.name, display_name=METRIC_PRC.name), 
                 hp.Metric(METRIC_ACC, display_name='Accuracy')],
    )

def train_model(hparams):
    model = Sequential([
        Conv2D(hparams[HP_NUM_UNITS_1], (3,3), activation=hparams[HP_ACT_FUNC], input_shape=input_shape),
        MaxPooling2D(2,2),
        Conv2D(hparams[HP_NUM_UNITS_2], (3,3), activation=hparams[HP_ACT_FUNC_2]),
        Flatten(),
        Dense(hparams[HP_NUM_UNITS_3], activation=hparams[HP_ACT_FUNC]),
        Dense(6, activation='softmax')
    ])
    model.compile(
        optimizer=hparams[HP_OPTIMIZER],
        loss='categorical_crossentropy',
        metrics=[METRIC_PRC, METRIC_ACC],
    )
    
    class_weights = class_weight.compute_class_weight(
               'balanced',
                np.unique(train_gen.classes), 
                train_gen.classes)

    train_class_weights = dict(enumerate(class_weights))
    
    model.fit(train_gen, 
              epochs=10,
              steps_per_epoch=STEP_SIZE_TRAIN,
              validation_data=val_gen,
              validation_steps=STEP_SIZE_VALID,
              class_weight=train_class_weights)
    
    _, prc, accuracy = model.evaluate(val_gen, steps=STEP_SIZE_VALID)
    return prc, accuracy

def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)
        prc, accuracy = train_model(hparams)
        tf.summary.scalar(METRIC_PRC.name, prc, step=1)
        tf.summary.scalar(METRIC_ACC, accuracy, step=1)

#%% Running hyperparameter tuning

session_num = 0

for num_units_1 in HP_NUM_UNITS_1.domain.values:
    for num_units_2 in HP_NUM_UNITS_2.domain.values:
        for num_units_3 in HP_NUM_UNITS_3.domain.values:
            for activation in HP_ACT_FUNC.domain.values:
                for activation_2 in HP_ACT_FUNC_2.domain.values:
                    for optimizer in HP_OPTIMIZER.domain.values:
                        tf.keras.backend.clear_session()
                        hparams = {
                            HP_NUM_UNITS_1: num_units_1,
                            HP_NUM_UNITS_2: num_units_2,
                            HP_NUM_UNITS_3: num_units_3,
                            HP_ACT_FUNC: activation,
                            HP_ACT_FUNC_2: activation_2,
                            HP_OPTIMIZER: optimizer,
                        }
                        run_name = "run-%d" % session_num
                        print('--- Starting trail: %s' % run_name)
                        print({h.name: hparams[h] for h in hparams})
                        run('logs/hparam_tuning_D/' + run_name, hparams)
                        session_num += 1

#%% final model (not decided yet)
modelD = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=input_shape, name='conv_1'),
    Conv2D(32, (3,3), activation='relu', name='conv_2'),
    MaxPooling2D(2,2),
    Conv2D(32, (3,3), activation='relu', name='conv_3'),
    Conv2D(64, (3,3), activation='relu', name='conv_4'),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(6, activation='softmax')
])

modelD.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

modelD.summary()

#%% Callbacks
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1)  # tensorboard --logdir ./logs
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
    patience=4) # reducing learning rate when val_loss doesn't improve for 4 epochs

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
