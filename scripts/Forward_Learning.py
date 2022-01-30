# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 01:37:23 2022

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
datagen = ImageDataGenerator(rescale=1/255)
datagen_test = ImageDataGenerator(rescale=1/255)
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
METRIC_KAPPA = tfa.metrics.CohenKappa(num_classes=2, name='kappa')
#%%
model_pfe = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(1, 1, 2560)),# flatten before fully connected part
    tf.keras.layers.Dense(32, activation='tanh'), 
    tf.keras.layers.Dense(64, activation='tanh'), 
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])
model_pfe.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy',METRIC_KAPPA],
    )
model_pfe.summary()

#%%
base_model = tf.keras.applications.EfficientNetB7(
    input_shape=(32, 32, 3), # refers to the shape we transfer from
    include_top=False, # cut off the head
    weights='imagenet', # pretrained on the ImageNet data
)
base_model.trainable = False # freeze the base model to not train it 
                             # not needed for pure feature extraction,
                             # but needed for the feature extraction 
                             # with multiple passes (as it is implemented here)
#%%
# Create new features
FE_train = base_model.predict(train_gen_CC) # Note the rescaling
FE_val = base_model.predict(val_gen_CC) # Note the rescaling
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
img_augmentation = Sequential(
    [
        tf.keras.layers.RandomRotation(factor=0.15),
        tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        tf.keras.layers.RandomFlip(),
        tf.keras.layers.RandomContrast(factor=0.1),
    ],
)
#%%
model_reloaded = tf.keras.models.load_model('saved_model.pb')
model_reloaded.summary()
#%%
def build_model():
    # create the base pre-trained model
    model = tf.keras.applications.efficientnet.EfficientNetB7(
        weights='imagenet', 
        input_shape=(32, 62, 3),
        include_top=False)

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = model.output
    x = tf.keras.layers.Flatten(input_shape=(1, 1, 2560))(x)
    x = tf.keras.layers.Dense(32, activation='tanh')(x)
    x = tf.keras.layers.Dense(64, activation='tanh')(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    # Compile
    model = tf.keras.Model(inputs=model.input, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss='binary_crossentropy',
        metrics=['accuracy',METRIC_KAPPA]
    )
    return model
#%%
strategy = tf.distribute.MirroredStrategy()
#%%
model1 = build_model()

epochs = 20  # @param {type: "slider", min:8, max:80}
hist = model.fit(train_gen_CC, validation_data=val_gen_CC, epochs=epochs, verbose=1)
#plot_hist(hist)

#%%
def unfreeze_model(model):
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-20:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy',METRIC_KAPPA]
    )


unfreeze_model(model)

tensorboard_callback = tf.keras.callbacks.TensorBoard('logs_exercise_nn_saveload/', histogram_freq=1)

epochs = 10  # @param {type: "slider", min:8, max:50}
hist = model.fit(train_gen_CC, validation_data=val_gen_CC, epochs=epochs, verbose=1, callbacks=[tensorboard_callback])
plot_hist(hist)

#%%
model.save('logs_exercise_nn_saveload/model_FWL')
#%%
model.fit(train_gen_CC, validation_data=val_gen_CC, epochs=5, verbose=1, callbacks=[tensorboard_callback])
model.save('logs_exercise_nn_saveload/model_FWL')
#%%
# create the base pre-trained model
base_model = tf.keras.applications.efficientnet.EfficientNetB7(
    weights='imagenet', 
    input_shape=(32, 32, 3),
    include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = tf.keras.layers.Flatten(input_shape=(1, 1, 2560))(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(1, activation='sigmoid')(x)

# this is the model we will train
model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional  layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='adam',   
              loss='binary_crossentropy',
              metrics=['accuracy',METRIC_KAPPA],)

# train the model on the new data for a few epochs
model.fit(train_gen_CC, validation_data=val_gen_CC, epochs=5, verbose=1)
#%%
# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)
#%%
# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), # use a low learning rate
    loss='binary_crossentropy',
    metrics=['accuracy',METRIC_KAPPA],
    )

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
tensorboard_callback = tf.keras.callbacks.TensorBoard('logs_exercise_nn_saveload/', histogram_freq=1)

model.fit(train_gen_CC, validation_data=val_gen_CC, epochs=5, verbose=1, callbacks=[tensorboard_callback])
model.save('logs_exercise_nn_saveload/model_FWL')



#%%
model_pfe = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(1, 1, 2560)),# flatten before fully connected part
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
model_pfe.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy',METRIC_KAPPA],
    )
model_pfe.summary()
model_pfe.fit(train_gen_CC, validation_data=val_gen_CC, epochs=5, verbose=1) # train for few epocs 
#%%
head = tf.keras.models.clone_model(model_pfe)
head.set_weights(model_pfe.get_weights())

base_model = tf.keras.applications.efficientnet.EfficientNetB7(
    input_shape=(32, 32, 3), # refers to the shape we transfer from
    include_top=False,       # cut off the head
    weights='imagenet',      # pretrained on the ImageNet data
)
base_model.training = False # to disable updating of means and 
                               # variances in batch norm layers
                               # This is in some cases very important, 
                               # but in this case it would still 
                               # work fine if we did not do it

for layer in base_model.layers[-20:]:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False             
  
model = tf.keras.models.Sequential([
    data_augmentation, 
    base_model, 
    head, 
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), # use a low learning rate
    loss='binary_crossentropy',
    metrics=['accuracy',METRIC_KAPPA],
    )
model.summary()
#%%
tensorboard_callback = tf.keras.callbacks.TensorBoard('logs_exercise_nn_saveload/', histogram_freq=1)

model.fit(train_gen_CC, validation_data=val_gen_CC, epochs=10, verbose=1, callbacks=[tensorboard_callback])
model.save('logs_exercise_nn_saveload/model_FWL')
#%%
model = tf.keras.models.load_model('exercise-nn-saveload')
#%%
head = tf.keras.models.clone_model(model_pfe)
head.set_weights(model_pfe.get_weights())

base_model = tf.keras.applications.efficientnet.EfficientNetB7(
    input_shape=(32, 32, 3), # refers to the shape we transfer from
    include_top=False,       # cut off the head
    weights='imagenet',      # pretrained on the ImageNet data
)
base_model.training = False # to disable updating of means and 
                               # variances in batch norm layers
                               # This is in some cases very important, 
                               # but in this case it would still 
                               # work fine if we did not do it

# Unfreeze the base_model. Note that it keeps running in inference mode
# since we passed `training=False` when calling it. This means that
# the batchnorm layers will not update their batch statistics.
# base_model.trainable = True  

for layer in base_model.layers[:200]: # layers after number 100 
    layer.trainable = False  
  
model = tf.keras.models.Sequential([
    data_augmentation, # can still use data augmentation now
    base_model, # the pre-trained part
    head, # the classifer we trained using feature extraction
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # use a low learning rate
    loss='binary_crossentropy',
    metrics=['accuracy',METRIC_KAPPA],
    )
model.summary()
#%%
model.summary()
#%%
history_ft = model.fit(train_gen_CC, validation_data=val_gen_CC, epochs=5, verbose=1)
#%%
base_model = tf.keras.applications.efficientnet.EfficientNetB7(
    weights="imagenet",  # Load weights pre-trained on ImageNet.
    input_shape=(32, 32, 3),
    include_top=False,
)  # Do not include the ImageNet classifier at the top.

# Freeze the base_model
base_model.trainable = False

# Create new model on top
inputs = tf.keras.Input(shape=(32, 32, 3))
x = data_augmentation(inputs)  # Apply random data augmentation

# The base model contains batchnorm layers. We want to keep them in inference mode
# when we unfreeze the base model for fine-tuning, so we make sure that the
# base_model is running in inference mode here.
x = base_model(x, training=False)
x = tf.keras.layers.Flatten(input_shape=(1, 1, 2560))(x)# flatten before fully connected part
x = tf.keras.layers.Dense(32, activation='tanh')(x)
x = tf.keras.layers.Dense(64, activation='tanh')(x) 
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs, outputs)

model.summary()
#%%
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), # use a low learning rate
    loss='binary_crossentropy',
    metrics=['accuracy',METRIC_KAPPA],
    )
#%%
epochs = 20
model.fit(train_gen_CC, validation_data=val_gen_CC, epochs=epochs, verbose=1)
#%%
# Unfreeze the base_model. Note that it keeps running in inference mode
# since we passed `training=False` when calling it. This means that
# the batchnorm layers will not update their batch statistics.
# This prevents the batchnorm layers from undoing all the training
# we've done so far.
base_model.trainable = True
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),  # Low learning rate
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy',METRIC_KAPPA],
)

epochs = 10
model.fit(train_gen_CC, validation_data=val_gen_CC, epochs=epochs, verbose=1)
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