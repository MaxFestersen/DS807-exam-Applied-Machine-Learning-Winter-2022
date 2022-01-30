#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 00:01:16 2022

@author: adernild
"""

#%% Importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
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

train_gen = datagen.flow_from_directory('data/split/Y/train', 
                                        batch_size=batch_size,
                                        shuffle=True,
                                        seed=1234,
                                        class_mode='categorical',
                                        color_mode='grayscale',
                                        target_size=(32, 62))

train_gen_aug = datagen_aug.flow_from_directory('data/split/Y/train', 
                                        batch_size=batch_size,
                                        shuffle=True,
                                        seed=1234,
                                        class_mode='categorical',
                                        color_mode='grayscale',
                                        target_size=(32, 62))

val_gen = datagen.flow_from_directory('data/split/Y/val', 
                                      batch_size=batch_size,
                                      shuffle=True,
                                      seed=1234,
                                      class_mode='categorical',
                                      color_mode='grayscale',
                                      target_size=(32, 62))

test_gen = datagen.flow_from_directory('data/split/Y/test',
                                       class_mode='categorical',
                                       color_mode='grayscale',
                                       target_size=(32, 62),
                                       shuffle=False)

input_shape=(32, 62, 1)
output=(11, 'softmax')
loss='categorical_crossentropy'

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
METRIC_KAPPA = tfa.metrics.CohenKappa(num_classes=11, name='kappa')
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

with tf.summary.create_file_writer('logs/hparam_tuning_D_reg_2').as_default():
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
                        run_('logs/hparam_tuning_Y_reg_2/' + run_name, hparams)
                        session_num += 1

#%%
METRIC_PRC = tf.keras.metrics.AUC(name='prc', curve='PR')
METRIC_ROC = tf.keras.metrics.AUC(name='ROC', curve='ROC')
METRIC_ACC = 'accuracy'
METRIC_KAPPA = tfa.metrics.CohenKappa(num_classes=11, name='kappa')

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
log_dir = "logs/fit_Y" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs/fit_Y', histogram_freq=1)  # tensorboard --logdir ./logs

checkpoint_filepath = './checkpoints/checkpoint-modelY'

modelCheckpoint = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True) # Saving best checkpoint

earlyStop = EarlyStopping(
    monitor='val_loss',
    patience=20) # stopping training when val_loss doesn't decrease in 15 epochs

reduceLR = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=4) # reducing learning rate when val_loss doesn't improve for 3 epochs

#%% Training no regularization
params = {'CONV_LAYER': 4, 
          'NUM_FILTS': 16, 
          'DENSE_LAYER': 2, 
          'NUM_UNITS': 64, 
          'ACTIVATION': 'relu', 
          'DROPOUT': 0.0}

modelY = final_model(params)

hist = modelY.fit(train_gen,
            steps_per_epoch=STEP_SIZE_TRAIN,
            validation_data=val_gen,
            validation_steps=STEP_SIZE_VALID,
            epochs=100,
            callbacks=[tensorboard_callback, modelCheckpoint, earlyStop])

#%% Evaluation
modelY.load_weights(checkpoint_filepath)
modelY.evaluate(test_gen) #accuracy 0.8738

#%%
modelY.save('models/modelY')

#%% Training with regularization
# best parameters
params = {'CONV_LAYER': 4, 
          'NUM_FILTS': 16, 
          'DENSE_LAYER': 2, 
          'NUM_UNITS': 64, 
          'ACTIVATION': 'relu', 
          'DROPOUT': 0.4}

# computing class weights
class_weights = class_weight.compute_class_weight(
           'balanced',
            np.unique(train_gen.classes), 
            train_gen.classes)

train_class_weights = dict(enumerate(class_weights))

modelY_reg = final_model(params)

hist_reg = modelY_reg.fit(train_gen,
            steps_per_epoch=STEP_SIZE_TRAIN,
            validation_data=val_gen,
            validation_steps=STEP_SIZE_VALID,
            epochs=100,
            callbacks=[tensorboard_callback, modelCheckpoint, earlyStop],
            class_weight=train_class_weights)

#%% Evaluation
modelY_reg.load_weights(checkpoint_filepath)
modelY_reg.evaluate(test_gen) #accuracy 0.9002

#%%
modelY_reg.save('models/modelY_reg')

#%% With data augmentation
modelY_aug = final_model(params)

hist_aug = modelY_aug.fit(train_gen_aug,
            steps_per_epoch=STEP_SIZE_TRAIN,
            validation_data=val_gen,
            validation_steps=STEP_SIZE_VALID,
            epochs=100,
            callbacks=[tensorboard_callback, modelCheckpoint, earlyStop],
            class_weight=train_class_weights)

#%% Evaluation
modelY_aug.load_weights(checkpoint_filepath)
modelY_aug.evaluate(test_gen) #accuracy 0.9323

#%%
modelY_aug.save('models/modelY_aug')

#%%
modelY = tf.keras.models.load_model('models/modelY')
modelY.evaluate(test_gen)

modelY_reg = tf.keras.models.load_model('models/modelY_reg')
modelY_reg.evaluate(test_gen)

modelY_aug = tf.keras.models.load_model('models/modelY_aug')
modelY_aug.evaluate(test_gen)

#%%
def plot_confusion_matrix(df_confusion, title='Confusion matrix'):
    sns.heatmap(df_confusion, annot=True, fmt='d', cmap='Blues')

y_test_hat = np.argmax(modelY.predict(test_gen), axis=1).flatten()
y_test_hat[y_test_hat == 2] = 11
y_test_hat[y_test_hat == 3] = 2
y_test_hat[y_test_hat == 4] = 3
y_test_hat[y_test_hat == 5] = 4
y_test_hat[y_test_hat == 6] = 5
y_test_hat[y_test_hat == 7] = 6
y_test_hat[y_test_hat == 8] = 7
y_test_hat[y_test_hat == 9] = 8
y_test_hat[y_test_hat == 10] = 9
y_test_hat[y_test_hat == 11] = 10

test_labels = test_gen.classes
test_labels[test_labels == 2] = 11
test_labels[test_labels == 3] = 2
test_labels[test_labels == 4] = 3
test_labels[test_labels == 5] = 4
test_labels[test_labels == 6] = 5
test_labels[test_labels == 7] = 6
test_labels[test_labels == 8] = 7
test_labels[test_labels == 9] = 8
test_labels[test_labels == 10] = 9
test_labels[test_labels == 11] = 10

df_confusion = pd.crosstab(test_gen.classes, y_test_hat, rownames=['Actual'], colnames=['Predicted'],dropna=False)

plot_confusion_matrix(df_confusion)
plt.savefig('plots/confusion_cnn_Y.png', dpi=300)

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

plot_hist(hist, 'val_accuracy', 'accuracy', 'plots/no_reg/modelY_acc.png')
plot_hist(hist, 'val_accuracy', 'prc', 'plots/no_reg/modelY_prc.png')
plot_hist(hist, 'val_accuracy', 'loss', 'plots/no_reg/modelY_loss.png')
plot_hist(hist, 'val_accuracy', 'kappa', 'plots/no_reg/modelY_kappa.png')

plot_hist(hist_reg, 'val_accuracy', 'accuracy', 'plots/reg/modelY_acc_reg.png')
plot_hist(hist_reg, 'val_accuracy', 'prc', 'plots/reg/modelY_prc_reg.png')
plot_hist(hist_reg, 'val_accuracy', 'loss', 'plots/reg/modelY_loss_reg.png')
plot_hist(hist_reg, 'val_accuracy', 'kappa', 'plots/reg/modelY_kappa_reg.png')
    
plot_hist(hist_aug, 'val_accuracy', 'accuracy', 'plots/aug/modelY_acc_aug.png')
plot_hist(hist_aug, 'val_accuracy', 'prc', 'plots/aug/modelY_prc_aug.png')
plot_hist(hist_aug, 'val_accuracy', 'loss', 'plots/aug/modelY_loss_aug.png')
plot_hist(hist_aug, 'val_accuracy', 'kappa', 'plots/aug/modelY_kappa_aug.png')

