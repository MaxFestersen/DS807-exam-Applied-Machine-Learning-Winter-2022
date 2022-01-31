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
import seaborn as sns
import matplotlib.pyplot as plt

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
modelCC = tf.keras.models.load_model('models/modelCC_reg')
modelD = tf.keras.models.load_model('models/modelD_aug')
modelY = tf.keras.models.load_model('models/modelY_aug')

modelCC.evaluate(test_gen_CC)
modelD.evaluate(test_gen_D)
#modelY.evaluate(test_gen_Y) #report graph error for some reason smh...

#%%
y_hat_CC = modelCC.predict(test_gen_CC)
y_hat_D = modelD.predict(test_gen_D)
y_hat_Y = modelY.predict(test_gen_Y)

y_hat_CC = np.where(y_hat_CC > 0.5, 1, 0).flatten()
y_hat_D = np.argmax(y_hat_D, axis=1).flatten()
y_hat_Y = np.argmax(y_hat_Y, axis=1).flatten()

# Due to a switch up in the labels from training to test set predicted labels has to be reassigned
y_hat_Y = np.select([y_hat_Y == 0,
                     y_hat_Y == 1,
                     y_hat_Y == 2, 
                     y_hat_Y== 3, 
                     y_hat_Y == 4, 
                     y_hat_Y == 5, 
                     y_hat_Y == 6,
                     y_hat_Y == 7,
                     y_hat_Y == 8,
                     y_hat_Y == 9,
                     y_hat_Y== 10], 
                    [0, 1, 10, 2, 3, 4, 5, 6, 7, 8, 9])

y_hat_D = np.select([y_hat_D == 0,
                     y_hat_D == 1,
                     y_hat_D == 2,
                     y_hat_D == 3,
                     y_hat_D == 4,
                     y_hat_D == 5],
                    [0,1,10,3,2,4])

df['CC_pred'] = y_hat_CC¨
df['D_pred'] = y_hat_D
df['Y_pred'] = y_hat_Y

#%%
true_pos = np.select([(df['CC'].astype(int) == df['CC_pred']) & (df['D'].astype(int) == df['D_pred']) & (df['Y'].astype(int) == df['Y_pred'])], [1], default=0)

true_pos_CC = np.select([(df['CC'].astype(int) == df['CC_pred'])], [1], default=0)
true_pos_D = np.select([(df['D'].astype(int) == df['D_pred'])], [1], default=0)
true_pos_Y = np.select([(df['Y'].astype(int) == df['Y_pred'])], [1], default=0)

test = pd.DataFrame({'CC': true_pos_CC, 'D': true_pos_D, 'Y': true_pos_Y})
test['total_score'] = (test['CC'] + test['D'] + test['Y'])/3
print(np.sum(true_pos)) # sequence accuracy
print(np.sum(test['total_score'])) # character accuracy

#%%
def plot_confusion_matrix(df_confusion, classes, title='Confusion matrix'):
    sns.heatmap(df_confusion, annot=True, fmt='d', cmap='Blues')
    plt.show()

confusion_CC = pd.crosstab(df['CC'].astype(int), y_hat_CC, rownames=['Actual'], colnames=['Predicted'],dropna=False)
confusion_D = pd.crosstab(df['D'].astype(int), y_hat_D, rownames=['Actual'], colnames=['Predicted'],dropna=False)
confusion_Y = pd.crosstab(df['Y'].astype(int), y_hat_Y, rownames=['Actual'], colnames=['Predicted'],dropna=False)
plot_confusion_matrix(confusion_CC, [0,1])
plot_confusion_matrix(confusion_D, [0,1,2,3,4,10])
plot_confusion_matrix(confusion_Y, [0,1,2,3,4,5,6,7,8,9,10])

#%%
if not os.path.exists('predictions/'):
    os.makedirs('predictions/')
np.save('predictions/y_hat_CC.npy', y_hat_CC)
np.save('predictions/y_hat_D.npy', y_hat_D)
np.save('predictions/y_hat_Y.npy', y_hat_Y)

#%%
df['CC_pred'] = y_hat_CC
df['D_pred'] = y_hat_D
df['Y_pred'] = y_hat_Y
