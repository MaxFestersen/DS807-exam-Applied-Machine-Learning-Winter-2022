#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 00:38:53 2022

@author: adernild
"""

#%% Importing libraries
import numpy as np
import tensorflow as tf
import pandas as pd
import tensorflow_addons as tfa
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout
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
datagen = ImageDataGenerator(rescale=1/255.0)

datagen_aug = ImageDataGenerator(rescale=1/255.0,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1)

datagen_test = ImageDataGenerator(rescale=1/255.0)
batch_size = 32

train_gen = datagen.flow_from_directory('data/split/CC/train', 
                                        batch_size=batch_size,
                                        shuffle=True,
                                        seed=1234,
                                        class_mode='binary',
                                        color_mode='grayscale',
                                        target_size=(32, 62))

train_gen_aug = datagen_aug.flow_from_directory('data/split/CC/train', 
                                        batch_size=batch_size,
                                        shuffle=True,
                                        seed=1234,
                                        class_mode='binary',
                                        color_mode='grayscale',
                                        target_size=(32, 62))

val_gen = datagen.flow_from_directory('data/split/CC/val', 
                                      batch_size=batch_size,
                                      shuffle=True,
                                      seed=1234,
                                      class_mode='binary',
                                      color_mode='grayscale',
                                      target_size=(32, 62))

test_gen = datagen.flow_from_directory('data/split/CC/test',
                                       batch_size=batch_size,
                                       class_mode='binary',
                                       color_mode='grayscale',
                                       target_size=(32, 62),
                                       shuffle=False)

input_shape=(32, 62, 1)
output=(1, 'sigmoid')
loss='binary_crossentropy'

STEP_SIZE_TRAIN=train_gen.n//train_gen.batch_size
STEP_SIZE_VALID=val_gen.n//val_gen.batch_size
STEP_SIZE_TEST=test_gen.n//test_gen.batch_size

#%% Setting up hyperparameter tuning 2
HP_CONV_LAYER = hp.HParam('conv_layer', hp.IntInterval(2,4))
HP_DENSE_LAYER = hp.HParam('dense_layer', hp.IntInterval(1,3))
HP_NUM_FILT = hp.HParam('num_filt', hp.Discrete([8,16]))
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([32,64,128]))
HP_ACTIVATION = hp.HParam('activation_func', hp.Discrete(['relu', 'tanh']))
HP_DROPOUT = hp.HParam("dropout", hp.RealInterval(0., 0.4))

METRIC_PRC = tf.keras.metrics.AUC(name='prc', curve='PR')
METRIC_ROC = tf.keras.metrics.AUC(name='ROC', curve='ROC')
METRIC_ACC = 'accuracy'
METRIC_KAPPA = tfa.metrics.CohenKappa(num_classes=2, name='kappa')

VAL_ACC = hp.Metric(
    "epoch_accuracy",
    group="validation",
    display_name="accuracy (val.)")

VAL_LOSS =hp.Metric(
    "epoch_loss",
    group="validation",
    display_name="loss (val.)")
TRAIN_ACC = hp.Metric(
    "batch_accuracy",
    group="train",
    display_name="accuracy (train)")
TRAIN_LOSS = hp.Metric(
    "batch_loss",
    group="train",
    display_name="loss (train)")

METRICS = [METRIC_PRC, 
           METRIC_ROC, 
           METRIC_ACC, 
           METRIC_KAPPA]

with tf.summary.create_file_writer('logs/hparam_tuning_CC_reg_2').as_default():
    hp.hparams_config(
        hparams=[HP_CONV_LAYER, HP_DENSE_LAYER,HP_NUM_FILT, HP_NUM_UNITS, HP_ACTIVATION, HP_DROPOUT],
        metrics=[hp.Metric(METRIC_PRC.name, display_name=METRIC_PRC.name), 
                 hp.Metric(METRIC_ACC, display_name='Accuracy'),
                 hp.Metric(METRIC_ROC.name, display_name=METRIC_ROC.name),
                 hp.Metric(METRIC_KAPPA.name, display_name=METRIC_KAPPA.name)],
    )

def create_model(hparams, logdir):
    
    model = Sequential()
    model.add(Input(input_shape))
    
    # add conv layers
    conv_filter = int(hparams[HP_NUM_FILT])
    for _ in range(hparams[HP_CONV_LAYER]):
        model.add(
            Conv2D(
                conv_filter,
                kernel_size=3,
                padding='same',
                activation=hparams[HP_ACTIVATION]
                )
            )
        model.add(MaxPooling2D(2, padding='same'))
        conv_filter *= 2
        
    model.add(Flatten())
    model.add(Dropout(hparams[HP_DROPOUT]))
    
    dense_neurons = int(hparams[HP_NUM_UNITS])
    for _ in range(hparams[HP_DENSE_LAYER]):
        model.add(Dense(dense_neurons, activation=hparams[HP_ACTIVATION]))
        dense_neurons *= 2
    
    model.add(Dense(output[0], activation=output[1]))
    
    model.compile(
        loss=loss,
        optimizer='adam',
        metrics=METRICS,
        )
    
    class_weights = class_weight.compute_class_weight(
               'balanced',
                np.unique(train_gen.classes), 
                train_gen.classes)

    train_class_weights = dict(enumerate(class_weights))

    callback = tf.keras.callbacks.TensorBoard(
        logdir
    )
    hparams_callback = hp.KerasCallback(logdir, hparams)
    
    model.fit(train_gen, 
              epochs=10,
              steps_per_epoch=STEP_SIZE_TRAIN,
              validation_data=val_gen,
              validation_steps=STEP_SIZE_VALID,
              class_weight=train_class_weights,
              callbacks=[callback, hparams_callback])
    
    _, prc, accuracy, roc, kappa = model.evaluate(val_gen, steps=STEP_SIZE_VALID)
    return prc, accuracy, roc, kappa

def run_(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)
        prc, accuracy, roc, kappa = create_model(hparams, run_dir)
        tf.summary.scalar(METRIC_PRC.name, prc, step=1)
        tf.summary.scalar(METRIC_ACC, accuracy, step=1)
        tf.summary.scalar(METRIC_ROC.name, roc, step=1)
        tf.summary.scalar(METRIC_KAPPA.name, kappa, step=1)

#%% Running hyperparameter tuning
session_num = 0

for conv_layer in range(HP_CONV_LAYER.domain.min_value, HP_CONV_LAYER.domain.max_value+1):
    for dense_layer in range(HP_DENSE_LAYER.domain.min_value, HP_DENSE_LAYER.domain.max_value+1):
        for filters in HP_NUM_FILT.domain.values:
            for neurons in HP_NUM_UNITS.domain.values:
                for activation in HP_ACTIVATION.domain.values:
                    for dropout_rate in tf.linspace(HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value, 3):
                        tf.keras.backend.clear_session()
                        hparams = {
                            HP_CONV_LAYER: conv_layer,
                            HP_DENSE_LAYER: dense_layer,
                            HP_NUM_FILT: filters,
                            HP_NUM_UNITS: neurons,
                            HP_ACTIVATION: activation,
                            HP_DROPOUT: float("%.2f"%float(dropout_rate)),
                            }
                        run_name = "run-%d" % session_num
                        print('--- Starting trail: %s' % run_name)
                        print({h.name: hparams[h] for h in hparams})
                        run_('logs/hparam_tuning_CC_reg_2/' + run_name, hparams)
                        session_num += 1

#%%
METRIC_PRC = tf.keras.metrics.AUC(name='prc', curve='PR')
METRIC_ROC = tf.keras.metrics.AUC(name='ROC', curve='ROC')
METRIC_ACC = 'accuracy'
METRIC_KAPPA = tfa.metrics.CohenKappa(num_classes=2, name='kappa')

METRICS = [METRIC_PRC, 
           METRIC_ROC, 
           METRIC_ACC, 
           METRIC_KAPPA]

def final_model(params):
    
    model = Sequential()
    model.add(Input(input_shape))
    
    # add conv layers
    conv_filter = params['NUM_FILTS']
    for _ in range(params['CONV_LAYER']):
        model.add(
            Conv2D(
                conv_filter,
                kernel_size=3,
                padding='same',
                activation=params['ACTIVATION']
                )
            )
        model.add(MaxPooling2D(2, padding='same'))
        conv_filter *= 2
        
    model.add(Flatten())
    model.add(Dropout(params['DROPOUT']))
    
    dense_neurons = params['NUM_UNITS']
    for _ in range(params['DENSE_LAYER']):
        model.add(Dense(dense_neurons, activation=params['ACTIVATION']))
        dense_neurons *= 2
    
    model.add(Dense(output[0], activation=output[1]))
    
    model.compile(
        loss=loss,
        optimizer='adam',
        metrics=METRICS,
        )
    return model

#%% Callbacks
log_dir = "logs/fit_CC/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)  # tensorboard --logdir ./logs

checkpoint_filepath = './checkpoints/checkpoint-modelCC'

modelCheckpoint = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_kappa',
    mode='max',
    save_best_only=True) # Saving best checkpoint

earlyStop = EarlyStopping(
    monitor='val_loss',
    patience=15) # stopping training when val_loss doesn't decrease in 15 epochs

reduceLR = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=4) # reducing learning rate when val_loss doesn't improve for 3 epochs

#%% Training no regularization
# best parameters
params = {'CONV_LAYER': 4, 
          'NUM_FILTS': 16, 
          'DENSE_LAYER': 2, 
          'NUM_UNITS': 32, 
          'ACTIVATION': 'tanh', 
          'DROPOUT': 0.0}

modelCC = final_model(params)

hist = modelCC.fit(train_gen,
            steps_per_epoch=STEP_SIZE_TRAIN,
            validation_data=val_gen,
            validation_steps=STEP_SIZE_VALID,
            epochs=100,
            callbacks=[tensorboard_callback, modelCheckpoint, earlyStop])

#%% Evaluation
modelCC.load_weights(checkpoint_filepath)
modelCC.evaluate(test_gen) #accuracy 0.9925

#%%
modelCC.save('models/modelCC')

#%% Training regularization
# best parameters
params = {'CONV_LAYER': 4, 
          'NUM_FILTS': 16, 
          'DENSE_LAYER': 2, 
          'NUM_UNITS': 32, 
          'ACTIVATION': 'tanh', 
          'DROPOUT': 0.4}

# computing class weights
class_weights = class_weight.compute_class_weight(
           'balanced',
            np.unique(train_gen.classes), 
            train_gen.classes)

train_class_weights = dict(enumerate(class_weights))

modelCC_reg = final_model(params)

hist_reg = modelCC_reg.fit(train_gen,
            steps_per_epoch=STEP_SIZE_TRAIN,
            validation_data=val_gen,
            validation_steps=STEP_SIZE_VALID,
            epochs=100,
            callbacks=[tensorboard_callback, modelCheckpoint, earlyStop],
            class_weight=train_class_weights)

#%% Evaluation
modelCC_reg.load_weights(checkpoint_filepath)
modelCC_reg.evaluate(test_gen) #accuracy 0.9925

#%%
modelCC.save('models/modelCC_reg')

#%% Training regularization + data augmentation

modelCC_aug = final_model(params)

hist_aug = modelCC.fit(train_gen_aug,
            steps_per_epoch=STEP_SIZE_TRAIN,
            validation_data=val_gen,
            validation_steps=STEP_SIZE_VALID,
            epochs=100,
            callbacks=[tensorboard_callback, modelCheckpoint, earlyStop],
            class_weight=train_class_weights)

#%% Evaluation
modelCC_aug.load_weights(checkpoint_filepath)
modelCC_aug.evaluate(test_gen) #accuracy 0.9908

#%%
modelCC_aug.save('models/modelCC_aug')

#%%
modelCC = tf.keras.models.load_model('models/modelCC')
modelCC.evaluate(test_gen)

modelCC = tf.keras.models.load_model('models/modelCC_reg')
modelCC.evaluate(test_gen)

modelCC_aug = tf.keras.models.load_model('models/modelCC_aug')
modelCC_aug.evaluate(test_gen)

#%%
def plot_confusion_matrix(df_confusion, title='Confusion matrix'):
    sns.heatmap(df_confusion, annot=True, fmt='d', cmap='Blues')

y_test_hat = np.where(modelCC.predict(test_gen) > 0.5, 1, 0).flatten()
df_confusion = pd.crosstab(test_gen.classes, y_test_hat, rownames=['Actual'], colnames=['Predicted'],dropna=False)

plot_confusion_matrix(df_confusion)
plt.savefig('plots/confusion_cnn_CC.png', dpi=300)
#%% Plotting model

def plot_hist(hist, stop_metric, metric, path):
    best_epoch = np.argmax(hist.history[stop_metric])
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_' + metric])
    plt.axvline(x=best_epoch)
    plt.title('model ' + metric)
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(path, dpi=300)
    plt.show()

plot_hist(hist, 'val_kappa', 'accuracy', 'plots/no_reg/modelCC_acc.png')
plot_hist(hist, 'val_kappa', 'prc', 'plots/no_reg/modelCC_prc.png')
plot_hist(hist, 'val_kappa', 'loss', 'plots/no_reg/modelCC_loss.png')
plot_hist(hist, 'val_kappa', 'kappa', 'plots/no_reg/modelCC_kappa.png')

plot_hist(hist_reg, 'val_kappa', 'accuracy', 'plots/reg/modelCC_acc_reg.png')
plot_hist(hist_reg, 'val_kappa', 'prc', 'plots/reg/modelCC_prc_reg.png')
plot_hist(hist_reg, 'val_kappa', 'loss', 'plots/reg/modelCC_loss_reg.png')
plot_hist(hist_reg, 'val_kappa', 'kappa', 'plots/reg/modelCC_kappa_reg.png')
    
plot_hist(hist_aug, 'val_kappa', 'accuracy', 'plots/aug/modelCC_acc_aug.png')
plot_hist(hist_aug, 'val_kappa', 'prc', 'plots/aug/modelCC_prc_aug.png')
plot_hist(hist_aug, 'val_kappa', 'loss', 'plots/aug/modelCC_loss_aug.png')
plot_hist(hist_aug, 'val_kappa', 'kappa', 'plots/aug/modelCC_kappa_aug.png')
