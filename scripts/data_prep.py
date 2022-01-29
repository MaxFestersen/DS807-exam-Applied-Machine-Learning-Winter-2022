# -*- coding: utf-8 -*-
"""
Data preparation
"""

#%% Importing libraries
import pandas as pd
import numpy as np
from numpy import save
import os
import matplotlib.pyplot as plt
import seaborn as sns
from shutil import copyfile
import splitfolders
from PIL import Image
import PIL.ImageOps
from PIL.ImageFilter import MedianFilter
from skimage.io import imread

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

#%% Plotting class distributions

fig, axs = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df['CC'], ax=axs[0])
sns.histplot(df['D'], ax=axs[1])
sns.histplot(df['Y'], ax=axs[2])

fig.savefig('plots/class_distributions.png', dpi=300)

#%% reassign 5,6,7,8,9 to class 10 in D
df.loc[df.D > 4, 'D'] = 10

#%% Prettyfying dataframe
df = df.rename(columns={0: "Name", 1: "Label"}) # Add name to label column

#%% Plotting class distributions
sns.histplot(df['CC'])
plt.figure()
sns.histplot(df['D'], bins=6, binwidth=1)
plt.savefig('plots/D_dist_new.png', dpi=300)
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

y_resize = int(y_avg/4)
x_resize = int(x_avg/4)

print(f'Smallest height: {np.array(y).min()}px, smallest width: {np.array(x).min()}px, largest height: {np.array(y).max()}px, largest width: {np.array(x).max()}px')
print(f'Average height: {y_avg}px, average width: {x_avg}px')

for f in files:
    image = Image.open(os.path.join('data/DIDA_1/' + f)).convert('L') # opens image and converts to grayscale
    image = PIL.ImageOps.invert(image).filter(MedianFilter(size=3)) # Inverts image and applies slight blur to denoise
    image = PIL.ImageOps.autocontrast(image, cutoff=(60,0)) # bumping contrast 
    bbox = image.point(lambda p: p > 45 and 255).getbbox() # finding bounding box in image
    image = image.crop(bbox) # cropping image to that bounding box
    image = PIL.ImageOps.pad(image, (x_resize, y_resize)) # up/down scales while keeping original aspect ratio
    if not os.path.exists('data/DIDA_2/'):
        os.makedirs('data/DIDA_2/') # making save directory if it doesn't exist
    image.save(os.path.join('data/DIDA_2/' + f), quality=100, subsampling=0) # saving image in highest quality

print(f'Images resized to height {y_resize}px, and width {x_resize}px')

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

#%% Generate test, train and validation vectores
def splitfolder_to_array(Categories, datadir):
    flat_data_arr=[] #input array
    target_arr=[] #output array
    for i in Categories:
        print(f'loading... category : {i}')
        path=os.path.join(datadir,i)
        for img in os.listdir(path):
            img_array=imread(os.path.join(path,img))
            flat_data_arr.append(img_array.flatten())
            target_arr.append(Categories.index(i))
        print(f'loaded category: {i} successfully')
    return np.array(flat_data_arr), np.array(target_arr);

X_train_CC, y_train_CC = splitfolder_to_array(Categories=['0','1'], datadir='data/split/CC/train')
X_test_CC, y_test_CC = splitfolder_to_array(Categories=['0','1'], datadir='data/split/CC/test')
X_val_CC, y_val_CC = splitfolder_to_array(Categories=['0','1'], datadir='data/split/CC/val')
print(X_train_CC.shape, X_test_CC.shape, y_train_CC.shape, y_test_CC.shape, X_val_CC.shape, y_val_CC.shape)

X_train_D, y_train_D = splitfolder_to_array(Categories=['0','1','2','3','4','10'], datadir='data/split/D/train')
X_test_D, y_test_D = splitfolder_to_array(Categories=['0','1','2','3','4','10'], datadir='data/split/D/test')
X_val_D, y_val_D = splitfolder_to_array(Categories=['0','1','2','3','4','10'], datadir='data/split/D/val')
print(X_train_D.shape, X_test_D.shape, y_train_D.shape, y_test_D.shape, X_val_D.shape, y_val_D.shape)

X_train_Y, y_train_Y = splitfolder_to_array(Categories=['0','1','2','3','4','5','6','7','8','9','10'], datadir='data/split/Y/train')
X_test_Y, y_test_Y = splitfolder_to_array(Categories=['0','1','2','3','4','5','6','7','8','9','10'], datadir='data/split/Y/test')
X_val_Y, y_val_Y = splitfolder_to_array(Categories=['0','1','2','3','4','5','6','7','8','9','10'], datadir='data/split/Y/val')
print(X_train_Y.shape, X_test_Y.shape, y_train_Y.shape, y_test_Y.shape, X_val_Y.shape, y_val_Y.shape)

#%% Save numpy array as npy file - Anders
#l1 = ['X_train_', 'y_train_', 'X_test_', 'y_test_', 'X_val_', 'y_val_']
#l2 = ['CC', 'D', 'Y']
#for i in l1:
#    for j in l2: 
#        a = f"'data/{i}{j}.npy'"
#        b = f'{i}{j}'
#        save(a,b)

#from numpy import asarray
#CC
save('data/X_train_CC.npy', X_train_CC)
save('data/y_train_CC.npy', y_train_CC)
save('data/X_test_CC.npy', X_test_CC)
save('data/y_test_CC.npy', y_test_CC)
save('data/X_val_CC.npy', X_val_CC)
save('data/y_val_CC.npy', y_val_CC)
#D
save('data/X_train_D.npy', X_train_D)
save('data/y_train_D.npy', y_train_D)
save('data/X_test_D.npy', X_test_D)
save('data/y_test_D.npy', y_test_D)
save('data/X_val_D.npy', X_val_D)
save('data/y_val_D.npy', y_val_D)
#Y
save('data/X_train_Y.npy', X_train_Y)
save('data/y_train_Y.npy', y_train_Y)
save('data/X_test_Y.npy', X_test_Y)
save('data/y_test_Y.npy', y_test_Y)
save('data/X_val_Y.npy', X_val_Y)
save('data/y_val_Y.npy', y_val_Y)

#%% Find common training images
#%% Find images and names in folders
files_all = []
files_by_category = {}
files_by_subcategory = {}
paths = ["data/split/CC/test", "data/split/D/test", "data/split/Y/test"]
for path in paths:
    for (dirpath, dirnames, filenames) in os.walk(path):
        if len(dirnames)>=2:
            continue
        #print(dirpath)
        files_all.extend(filenames)
        cat = dirpath.split("data/split/")[1]
        cat = cat.split("/")[0]
        subcat = dirpath.split("\\")[1]
        cat_string = cat + " " + subcat
        if cat in files_by_category:
            files_by_category[cat] = files_by_category[cat] + filenames
        else:
            files_by_category[cat] = filenames
        if cat_string in files_by_category:
            files_by_subcategory[cat_string] = files_by_subcategory[cat_string] + filenames
        else:
            files_by_subcategory[cat_string] = filenames

#%% Find matches
print(set(files_by_category["CC"]) & set(files_by_category["D"]) & set(files_by_category["Y"]))
