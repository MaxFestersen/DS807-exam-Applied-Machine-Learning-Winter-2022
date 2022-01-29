#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 14:51:42 2022

@author: adernild
"""
from PIL import Image
import PIL.ImageOps
from PIL.ImageFilter import MedianFilter
import matplotlib.pyplot as plt
import os
import numpy as np


files = os.listdir('data/DIDA_1')

y = []
x = []

for f in files:
    image = Image.open(os.path.join('data/DIDA_1/' + f))
    width, height = image.size
    y.append(height)
    x.append(width)

y_avg = int(np.sum(y) / len(y))
x_avg = int(np.sum(x) / len(x))

y_resize = int(y_avg/4)
x_resize = int(x_avg/4)

print(f'Smallest height: {np.array(y).min()}px, smallest width: {np.array(x).min()}px, largest height: {np.array(y).max()}px, largest width: {np.array(x).max()}px')
print(f'Average height: {y_avg}px, average width: {x_avg}px')
print(f'Images resized to height {y_resize}px, and width {x_resize}px')
#%%
def preprocess_img(image):
    image = image.convert('L') # opens image and converts to grayscale
    image = PIL.ImageOps.invert(image).filter(MedianFilter(size=3)) # Inverts image and applies slight blur to denoise
    image = PIL.ImageOps.autocontrast(image, cutoff=(60,0)) # bumping contrast
    return image
    
def resize_img(img, size=(62,32)):
    image = img
    bbox = image.point(lambda p: p > 45 and 255).getbbox() # finding bounding box in image
    image = image.crop(bbox) # cropping image to that bounding box
    image = PIL.ImageOps.pad(image, (size[0], size[1])) # up/down scales while keeping original aspect ratio
    return image

image = Image.open('data/DIDA_1/129.jpg')
image_2 = Image.open('data/DIDA_1/127.jpg')

img = preprocess_img(image)
#img = resize_img(img)


plt.imshow(img, cmap='gray')

fig, axs = plt.subplots(2,2)
axs[0,0].imshow(image)
axs[0,1].imshow(image_2)
axs[1,0].imshow(preprocess_img(image), cmap='gray')
axs[1,1].imshow(preprocess_img(image_2), cmap='gray')

fig.savefig('plots/image_preprocessing.png', dpi=300)

image = Image.open('data/DIDA_1/23.jpg')
image_2 = Image.open('data/DIDA_1/26.jpg')


fig, axs = plt.subplots(2,2)
axs[0,0].imshow(image)
axs[0,1].imshow(image_2)
axs[1,0].imshow(resize_img(preprocess_img(image)), cmap='gray')
axs[1,1].imshow(resize_img(preprocess_img(image_2)), cmap='gray')
fig.savefig('plots/image_resizing.png', dpi=300)
