# -*- coding: utf-8 -*-
"""
Data preparation
"""

#%% Importing libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from shutil import copyfile
import splitfolders
from PIL import Image

# Set path to parrent location of current file
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
os.chdir("../")

#%% Importing data and encoding new colums

df = pd.read_csv("data/DIDA_12000_String_Digit_Labels.csv", header=None)

df['path'] = "DIDA_2/" + df[0].astype(str) + ".jpg"
df['CC'] = np.select([(df[1]<1900) & (df[1]>1799)], [0], default=1)
df['D'] = np.select([(df[1].astype(str).str.len()>1) & (df[1].astype(str).str.len()<5)], [df[1].astype(str).str[-2].astype(int)], default=10)
df['Y'] = np.select([(df[1].astype(str).str.len()>1) & (df[1].astype(str).str.len()<5)], [df[1].astype(str).str[-1].astype(int)], default=10)

#%% Prettyfying dataframe
df = df.rename(columns={0: "Name", 1: "Label"}) # Add name to label column

#%% Plotting class distributions
sns.histplot(df['CC'])
plt.figure()
sns.histplot(df['D'])
plt.figure()
sns.histplot(df['Y'])

#%% Saving new .csv
df.to_csv("data/data.csv", index=False)

#%% Finding average image size and resizing images to this size
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

print(f'Smallest height: {np.array(y).min()}px, smallest width: {np.array(x).min()}px, largest height: {np.array(y).max()}px, largest width: {np.array(x).max()}px')
print(f'Average height: {y_avg}px, average width: {x_avg}px')

for f in files:
    image = Image.open(os.path.join('data/DIDA_1/' + f))
    image = image.resize((x_avg, y_avg), Image.NEAREST)
    if not os.path.exists('data/DIDA_2/'):
        os.makedirs('data/DIDA_2/')
    image.save(os.path.join('data/DIDA_2/' + f), quality=100, subsampling=0)

print(f'Images resized to height {y_avg}px, and width {x_avg}px')

#%% Copying files to CC folder
for i in range(0, len(df)):
    src = os.path.join('data/', df.iloc[i, 2])
    dst = os.path.join('data/CC/', str(df.iloc[i, 3]) + "/", str(df.iloc[i, 0]) + ".jpg")
    if not os.path.exists(os.path.dirname(dst)):
        os.makedirs(os.path.dirname(dst))
    copyfile(src, dst)

#%% Copying files to D folder
for i in range(0, len(df)):
    src = os.path.join('data/', df.iloc[i, 2])
    dst = os.path.join('data/D/', str(df.iloc[i, 4]) + "/", str(df.iloc[i, 0]) + ".jpg")
    if not os.path.exists(os.path.dirname(dst)):
        os.makedirs(os.path.dirname(dst))
    copyfile(src, dst)
    
#%% Copying files to Y folder
for i in range(0, len(df)):
    src = os.path.join('data/', df.iloc[i, 2])
    dst = os.path.join('data/Y/', str(df.iloc[i, 5]) + "/", str(df.iloc[i, 0]) + ".jpg")
    if not os.path.exists(os.path.dirname(dst)):
        os.makedirs(os.path.dirname(dst))
    copyfile(src, dst)

#%% Splitting folders to train/test/val
splitfolders.ratio("data/CC/", output="data/split/CC/", seed=1337, ratio=(0.8, 0.1, 0.1))
splitfolders.ratio("data/D/", output="data/split/D/", seed=1337, ratio=(0.8, 0.1, 0.1))
splitfolders.ratio("data/Y/", output="data/split/Y/", seed=1337, ratio=(0.8, 0.1, 0.1))
