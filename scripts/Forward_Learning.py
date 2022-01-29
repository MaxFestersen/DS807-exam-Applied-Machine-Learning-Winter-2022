# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 01:37:23 2022

@author: A
"""
#%% Importing libraries
import numpy as np
import tensorflow as tf
import pandas as pd
#import tensorflow_addons as tfa
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

datagen = ImageDataGenerator(rescale=1/255.0)
datagen_test = ImageDataGenerator(rescale=1/255.0)
batch_size = 1

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
train_gen_D = datagen.flow_from_directory('data/split/D/train', 
                                        batch_size=batch_size,
                                        shuffle=True,
                                        seed=1234,
                                        class_mode='binary',
                                        color_mode='grayscale',
                                        target_size=(32, 62))

train_gen_D = datagen.flow_from_directory('data/split/D/val', 
                                      batch_size=batch_size,
                                      shuffle=True,
                                      seed=1234,
                                      class_mode='binary',
                                      color_mode='grayscale',
                                      target_size=(32, 62))

train_gen_D = datagen.flow_from_directory('data/split/D/test',
                                       batch_size=batch_size,
                                       class_mode='binary',
                                       color_mode='grayscale',
                                       target_size=(32, 62),
                                       shuffle=False)

#%%
train_gen_Y = datagen.flow_from_directory('data/split/Y/train', 
                                        batch_size=batch_size,
                                        shuffle=True,
                                        seed=1234,
                                        class_mode='binary',
                                        color_mode='grayscale',
                                        target_size=(32, 62))

train_gen_Y = datagen.flow_from_directory('data/split/Y/val', 
                                      batch_size=batch_size,
                                      shuffle=True,
                                      seed=1234,
                                      class_mode='binary',
                                      color_mode='grayscale',
                                      target_size=(32, 62))

train_gen_Y = datagen.flow_from_directory('data/split/Y/test',
                                       batch_size=batch_size,
                                       class_mode='binary',
                                       color_mode='grayscale',
                                       target_size=(32, 62),
                                       shuffle=False)

input_shape=(32, 62, 1)
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
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.1, 
                                                              input_shape=(32, 32, 3)),
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
    ])

#%%
base_model = tf.keras.applications.efficientnet.EfficientNetB7(
    input_shape=(32, 32, 3), # refers to the shape we transfer from - not the shape we use, which is 32x62
    include_top=False, # cut off the head
    weights='imagenet', # pretrained on the ImageNet data
)
base_model.trainable = False # freeze the base model to not train it. not needed for pure feature extraction,
# but needed for the feature extraction with multiple passes (as it is implemented here - NOT always needed)
#%%
model_pfe = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(1, 1, 2560)),# flatten before fully connected part
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
    ])
model_pfe.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
    )
model_pfe.summary()
#%%
head_ft = tf.keras.models.clone_model(model_pfe)
head_ft.set_weights(model_pfe.get_weights())

base_model_ft = tf.keras.applications.efficientnet.EfficientNetB7(
    input_shape=(32, 32, 3), # refers to the shape we transfer from
    include_top=False,       # cut off the head
    weights='imagenet',      # pretrained on the ImageNet data
)
base_model_ft.training = False # to disable updating of means and 
                               # variances in batch norm layers
                               # This is in some cases very important, 
                               # but in this case it would still 
                               # work fine if we did not do it

#for layer in base_model_ft.layers[:100]: # layers after number 100 
#    layer.trainable = False              #in base_model now trainable
    
model_ft = tf.keras.models.Sequential([
    data_augmentation, # can still use data augmentation now
    base_model_ft, # the pre-trained part
    head_ft, # the classifer we trained using feature extraction
])

model_ft.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), # use a low learning rate
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
    )
#%%
model_ft.summary()
#%%
history_ft = model_ft.fit(train_gen_CC, validation_data=(test_images, test_labels), epochs=15, verbose=1)

#%% Excited layers
example_image = train_images[0:1].copy()
plt.imshow(example_image[0])
plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1); 
    plt.axis('off'); 
    plt.imshow(model.get_layer('conv2d')(example_image)[0, :, :, i])
#%% Heatmap
def get_heatmap(category, start_image):
    frog_output     = model.get_layer('dense_1').output # output layer
    last_conv_layer = model.get_layer('conv2d_2').output # deep convolution 
                                                         # we could use something else
    submodel = tf.keras.models.Model([model.inputs], [frog_output, last_conv_layer])

    input_img_data = start_image.copy()
    input_img_data = tf.Variable(tf.cast(input_img_data, tf.float32))

    with tf.GradientTape() as tape:
        outputs_class, outputs_conv = submodel(input_img_data)
        loss_value                  = tf.reduce_mean(outputs_class[:, category])

    grads = tape.gradient(loss_value, outputs_conv)

    cast_outputs_conv = tf.cast(outputs_conv > 0, "float32")
    cast_grads        = tf.cast(grads > 0, "float32")
    guided_grads      = cast_outputs_conv * cast_grads * grads
    outputs_conv      = outputs_conv[0]
    guided_grads      = guided_grads[0]
    
    weights           = tf.reduce_mean(guided_grads, axis=(0, 1))
    cam               = tf.reduce_sum(tf.multiply(weights, outputs_conv), axis=-1)
    
    return cam
#%%
from PIL import Image
cmap = plt.get_cmap('jet')

def create_heatmap(idx):
    category, image = train_labels[idx], train_images[idx:(idx + 1)]

    heatmap = get_heatmap(category, image)
    heatmap = heatmap.numpy()
    heatmap = Image.fromarray(heatmap)
    heatmap = heatmap.resize((32, 32), Image.ANTIALIAS) # upscale
    heatmap = np.array(heatmap) # back to numpy array
    heatmap = (heatmap / heatmap.max()) # to [0, 1]    
    heatmap = cmap(heatmap)
    heatmap = np.delete(heatmap, 3, 2)

    overlayed_heatmap = 0.6 * image[0] + 0.4 * heatmap
    
    return image[0], heatmap, overlayed_heatmap

def plot_heatmap():
    plt.figure(figsize=(10, 10))
    for i in range(3):
        images = create_heatmap(i)
        for j in range(3):
            ax = plt.subplot(3, 3, i * 3 + 1 + j); plt.axis('off'); plt.imshow(images[j])
    plt.show()
#%%
plot_heatmap()