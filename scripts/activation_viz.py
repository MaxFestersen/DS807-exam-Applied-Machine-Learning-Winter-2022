#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 23:49:03 2022

@author: adernild
"""
import PIL
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from numpy import expand_dims
import os

# Set path to parrent location of current file
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
os.chdir("../")


model = load_model('models/modelD_aug')

img = PIL.Image.open('data/DIDA_2/46.jpg')
img = img_to_array(img)
img = expand_dims(img, axis=0)

plt.figure(figsize=(10, 10))
for i in range(16):
    ax = plt.subplot(4, 4, i + 1); 
    plt.axis('off'); 
    plt.imshow(model.get_layer('conv2d_50')(img)[0, :, :, i])

plt.savefig('plots/activation_viz.png', dpi=300)

