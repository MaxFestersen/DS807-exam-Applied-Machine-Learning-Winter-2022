# -*- coding: utf-8 -*-
"""
Moving random subset of new images
"""

#%% Importing libraries
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import seaborn as sns
import shutil
import pandas as pd
from PIL import Image
import PIL.ImageOps
from PIL.ImageFilter import MedianFilter
# Set path to parrent location of current file
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
os.chdir("../")

#%% Splitting folders to train/test/val
random.seed(1377)
files = os.listdir("data/Digit_String_2 (published)")
no_of_files = 500
os.mkdir("data/newsplit/")

random.shuffle(files)
files = files[:no_of_files]

for name in files:
    print(name)
    shutil.copyfile(os.path.join("data/Digit_String_2 (published)", name), os.path.join("data/newsplit/", name))

#%% Get filenames from folder and sort them
files = []

for (dirpath, dirnames, filenames) in os.walk("data/newsplit/"):
    files.extend(filenames)
    break
#print(files)

#%% Importing data and encoding new colums
df_n_labels = pd.read_csv("labels/Digit_String_2_labels.csv", header=None)

#%% Remove Na
df_n_labels[1] = pd.to_numeric(df_n_labels[1], errors='coerce')

# Get nan values
df_n_labels_nan = df_n_labels[df_n_labels[1].isnull()]

# Remove nan values from labels
df_n_labels = df_n_labels.dropna()
df_n_labels[1] = df_n_labels[1].astype(int)

# Remove images that could not be read from the folder
for file in df_n_labels_nan[0].astype(str):
    os.remove("data/newsplit/" + file + ".jpg")

#%% CC-D-Y moddeling strategy implementation
df_n_labels['path'] = "data/newsplit/" + df_n_labels[0].astype(str) + ".jpg"
df_n_labels['CC'] = np.select([(df_n_labels[1]<1900) & (df_n_labels[1]>1799)], [0], default=1)
df_n_labels['D'] = np.select([(df_n_labels[1].astype(str).str.len()>1) & (df_n_labels[1].astype(str).str.len()<5)], [df_n_labels[1].astype(str).str[-2].astype(int)], default=10)
df_n_labels['Y'] = np.select([(df_n_labels[1].astype(str).str.len()>1) & (df_n_labels[1].astype(str).str.len()<5)], [df_n_labels[1].astype(str).str[-1].astype(int)], default=10)

#%% reassign 5,6,7,8,9 to class 10 in D
df_n_labels.loc[df_n_labels.D > 4, 'D'] = 10

#%% Prettyfying dataframe
df_n_labels = df_n_labels.rename(columns={0: "Name", 1: "Label"}) # Add name to label column

#%% Plotting class distributions
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df_n_labels['CC'], ax=axs[0])
sns.histplot(df_n_labels['D'], ax=axs[1])
sns.histplot(df_n_labels['Y'], ax=axs[2])

fig.savefig('plots/class_distributions_test2.png', dpi=300)

#%% Saving new .csv
df_n_labels.to_csv("labels/Digit_String_2_labels_formatted.csv", index=False)

#%% Rezising
y_resize = 130
x_resize  = 250

for path in df_n_labels['path']:
    image = Image.open(path).convert('L') # opens image and converts to grayscale
    image = PIL.ImageOps.invert(image).filter(MedianFilter(size=3)) # Inverts image and applies slight blur to denoise
    image = PIL.ImageOps.autocontrast(image, cutoff=(60,0)) # bumping contrast 
    bbox = image.point(lambda p: p > 45 and 255).getbbox() # finding bounding box in image
    image = image.crop(bbox) # cropping image to that bounding box
    image = PIL.ImageOps.pad(image, (x_resize, y_resize)) # up/down scales while keeping original aspect ratio
    image.save(path, quality=100, subsampling=0) # saving image in highest quality
