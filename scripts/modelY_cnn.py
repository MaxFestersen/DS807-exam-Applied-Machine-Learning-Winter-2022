#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 10:15:55 2022

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
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
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
datagen_test = ImageDataGenerator(rescale=1/255.0)
batch_size = 32

train_gen = datagen.flow_from_directory('data/split/Y/train', 
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

#%% Setting up hyperparameter tuning
HP_NUM_FILTS = hp.HParam('num_filts', hp.Discrete([8,16,32]))
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([128,256,512]))
HP_ACT_FUNC = hp.HParam('activation_func', hp.Discrete(['relu', 'tanh']))
HP_ACT_FUNC_2 = hp.HParam('activation_func_2', hp.Discrete(['relu', 'tanh']))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
HP_DROPOUT = hp.HParam("dropout", hp.RealInterval(0., 0.4))

METRIC_PRC = tf.keras.metrics.AUC(name='prc', curve='PR')
METRIC_ROC = tf.keras.metrics.AUC(name='ROC', curve='ROC')
METRIC_ACC = 'accuracy'

with tf.summary.create_file_writer('logs/hparam_tuning_Y_reg').as_default():
    hp.hparams_config(
        hparams=[HP_NUM_FILTS, HP_NUM_UNITS, HP_ACT_FUNC, HP_ACT_FUNC_2, HP_OPTIMIZER],
        metrics=[hp.Metric(METRIC_PRC.name, display_name=METRIC_PRC.name), 
                 hp.Metric(METRIC_ACC, display_name='Accuracy'),
                 hp.Metric(METRIC_ROC.name, display_name=METRIC_ROC.name)],
    )

def train_model(hparams):
    num_filts = int(hparams[HP_NUM_FILTS])
    model = Sequential([
        Conv2D(num_filts, (3,3), activation=hparams[HP_ACT_FUNC], input_shape=input_shape),
        MaxPooling2D(2,2),
        Conv2D(num_filts*2, (3,3), activation=hparams[HP_ACT_FUNC_2]),
        Flatten(),
        Dense(hparams[HP_NUM_UNITS], activation=hparams[HP_ACT_FUNC]),
        Dropout(hparams[HP_DROPOUT]),
        Dense(11, activation='softmax')
    ])
    model.compile(
        optimizer=hparams[HP_OPTIMIZER],
        loss='categorical_crossentropy',
        metrics=[METRIC_PRC, METRIC_ACC, METRIC_ROC],
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
    
    _, prc, accuracy, roc = model.evaluate(val_gen, steps=STEP_SIZE_VALID)
    return prc, accuracy, roc

def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)
        prc, accuracy, roc = train_model(hparams)
        tf.summary.scalar(METRIC_PRC.name, prc, step=1)
        tf.summary.scalar(METRIC_ACC, accuracy, step=1)
        tf.summary.scalar(METRIC_ROC.name, roc, step=1)

#%% Running hyperparameter tuning

session_num = 0

for num_filts in HP_NUM_FILTS.domain.values:
    for num_units in HP_NUM_UNITS.domain.values:
        for activation in HP_ACT_FUNC.domain.values:
            for activation_2 in HP_ACT_FUNC_2.domain.values:
                for optimizer in HP_OPTIMIZER.domain.values:
                    for dropout_rate in tf.linspace(HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value, 3):
                        tf.keras.backend.clear_session()
                        hparams = {
                            HP_NUM_FILTS: num_filts,
                            HP_NUM_UNITS: num_units,
                            HP_ACT_FUNC: activation,
                            HP_ACT_FUNC_2: activation_2,
                            HP_OPTIMIZER: optimizer,
                            HP_DROPOUT: float("%.2f"%float(dropout_rate)),
                        }
                        run_name = "run-%d" % session_num
                        print('--- Starting trail: %s' % run_name)
                        print({h.name: hparams[h] for h in hparams})
                        run('logs/hparam_tuning_Y_reg/' + run_name, hparams)
                        session_num += 1

#%% final model (not decided yet)
def final_model(num_filts, num_units, act_func_1, act_func_2, dropout):
    modelY = Sequential([
        Conv2D(num_filts, (3,3), activation=act_func_1, input_shape=input_shape, name='conv_1'),
        MaxPooling2D(2,2),
        Conv2D(num_filts*2, (3,3), activation=act_func_2, name='conv_2'),
        Flatten(),
        Dense(num_units, activation=act_func_1),
        Dropout(dropout),
        Dense(11, activation='sigmoid')
    ])

    METRICS = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
        tfa.metrics.CohenKappa(num_classes=11, name='kappa'),
        tf.keras.metrics.AUC(name='ROC', curve='ROC')
    ]

    modelY.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=METRICS)

    return modelY

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

#%% Training
# computing class weights
class_weights = class_weight.compute_class_weight(
           'balanced',
            np.unique(train_gen.classes), 
            train_gen.classes)

train_class_weights = dict(enumerate(class_weights))

STEP_SIZE_TRAIN=train_gen.n//train_gen.batch_size
STEP_SIZE_VALID=val_gen.n//val_gen.batch_size

modelY = final_model(32, 32, 256, 'relu', 'tanh')

hist = modelY.fit(train_gen,
            steps_per_epoch=STEP_SIZE_TRAIN,
            validation_data=val_gen,
            validation_steps=STEP_SIZE_VALID,
            epochs=100,
            callbacks=[tensorboard_callback, modelCheckpoint, earlyStop])

#%% Evaluation
modelY.load_weights(checkpoint_filepath)
modelY.evaluate(test_gen) #accuracy 0.8422

#%%
modelY.save('models/modelY')

#%%

modelY = tf.keras.models.load_model('models/modelY')
modelY.evaluate(test_gen)

#%%
def plot_confusion_matrix(df_confusion, title='Confusion matrix'):
    sns.heatmap(df_confusion, annot=True, fmt='d', cmap='Blues')

y_test_hat = np.argmax(modelY.predict(test_gen), axis=1).flatten()
df_confusion = pd.crosstab(test_gen.classes, y_test_hat, rownames=['Actual'], colnames=['Predicted'],dropna=False)

plot_confusion_matrix(df_confusion)

#%% Plotting model
best_epoch = np.argmax(hist.history['val_accuracy'])

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.axvline(x=best_epoch)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('plots/modelY_acc.png', dpi=300)
plt.show()

plt.plot(hist.history['prc'])
plt.plot(hist.history['val_prc'])
plt.axvline(x=best_epoch)
plt.title('model PRc')
plt.ylabel('Precision-Recall curve')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('plots/modelY_prc.png', dpi=300)
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.axvline(x=best_epoch)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('plots/modelY_loss.png', dpi=300)
plt.show()

plt.plot(hist.history['kappa'])
plt.plot(hist.history['val_kappa'])
plt.axvline(x=best_epoch)
plt.title('model kappa')
plt.ylabel('kappa')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('plots/modelY_kappa.png', dpi=300)
plt.show()