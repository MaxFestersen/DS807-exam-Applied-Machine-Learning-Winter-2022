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
    
    
#%%
for _ in range(5):
    img, label = dir_It.next()
    print(img.shape)   #  (1,256,256,3)
    plt.imshow(img[0])
    plt.show()
    
#%%    

img, label = test_gen_CC.next()
image = plt.imshow(img[0])
plt.show()    
 
#%%
img, label = test_gen_CC.next()
    #print(img.shape)   #  (1,256,256,3)
    plt.imshow(img[0])
    plt.show()   
#%%    
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(15,15))
    for i in range(4):
         image = next(aug_iter)[0].astype('uint8')
        
         image = np.squeeze(image)
        
         ax[i].imshow(image, cmap='Blues')
         ax[i].axis('off')
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
model = build_model()

epochs = 20  # @param {type: "slider", min:8, max:80}
hist = model.fit(train_gen_CC, validation_data=val_gen_CC, epochs=epochs, verbose=1)
plot_hist(hist)
#%%
KERAS_MODEL_NAME = "FWL_Model.hdf5"
#model.save('exercise-nn-saveload')
keras.models.save_model(model, KERAS_MODEL_NAME)
#%%
def unfreeze_model(model):
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-2000:]:
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
joblib.dump(model, 'data/model_FWL.pkl')
model.save('model_FW')
model2 = tf.keras.models.load_model('logs_exercise_nn_saveload')
#%%
plot_loss(hist)
y_hat = model.predict(test_gen_CC)
y_hat_bin = np.where(y_hat > 0.5, 1, 0)
#%%
model.save('logs_exercise_nn_saveload/model_FWL')
#%%
model.fit(train_gen_CC, validation_data=val_gen_CC, epochs=5, verbose=1, callbacks=[tensorboard_callback])
model.save('logs_exercise_nn_saveload/model_FWL')

#%% Load model 
model = tf.keras.models.load_model('FWL_Model.hdf5')

#%% Excited layers
import PIL
from tensorflow.keras.preprocessing.image import img_to_array
from numpy import expand_dims
img = PIL.Image.open('data/DIDA_2/46.jpg')
img = img_to_array(img)
img = expand_dims(img, axis=0)
plt.figure(figsize=(10, 10))
for i in range(16):
    ax = plt.subplot(4, 4, i + 1); 
    plt.axis('off'); 
    plt.imshow(model.get_layer(last_conv_layer_name).output)(img)[0, :, :, i]
    #model.get_layer(last_conv_layer_name).output
    
    
#%%
from skimage import io
from PIL import Image

img = io.imread('data/DIDA_2/46.jpg')
display(img)
img = Image.open("data/DIDA_2/46.jpg")
#%% 
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#%%
model_builder = keras.applications.xception.Xception
img_size = (32, 62)
preprocess_input = keras.applications.xception.preprocess_input
decode_predictions = keras.applications.xception.decode_predictions

last_conv_layer_name = "block14_sepconv2_act"

# The local path to our target image
imgage = Image.open("data/DIDA_2/46.jpg")

display(imgage)
#%%
def get_img_array(img_path):
    # `img` is a PIL image of size 299x299
    img = img_path
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()
#%%
# Prepare image
img_array = preprocess_input(get_img_array(img_path))

# Make model
model1 = model

# Remove last layer
#model.layers[-1].activation = None

# Print what the top predicted class is
preds = model1.predict(img_array)
print("Predicted:", decode_predictions(preds, top=1)[0])

# Generate class activation heatmap
heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

# Display heatmap
plt.matshow(heatmap)
plt.show()

#%%
#%%
def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    display(Image(cam_path))


save_and_display_gradcam(img_path, heatmap)