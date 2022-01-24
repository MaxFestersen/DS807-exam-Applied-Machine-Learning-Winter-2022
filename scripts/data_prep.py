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
os.chdir("../")

#%% Importing data and encoding new colums

df = pd.read_csv("data/DIDA_12000_String_Digit_Labels.csv", header=None)

df['path'] = "DIDA_1/" + df[0].astype(str) + ".jpg"
df['CC'] = np.select([(df[1]<1900) & (df[1]>1799)], [0], default=1)
df['D'] = np.select([(df[1].astype(str).str.len()>1) & (df[1].astype(str).str.len()<5)], [df[1].astype(str).str[-2].astype(int)], default=10)
df['Y'] = np.select([(df[1].astype(str).str.len()>1) & (df[1].astype(str).str.len()<5)], [df[1].astype(str).str[-1].astype(int)], default=10)

#%% Plotting class distributions

sns.histplot(df['CC'])
plt.figure()
sns.histplot(df['D'])
plt.figure()
sns.histplot(df['Y'])

#%% Saving new .csv
df.to_csv("data/data.csv", index=False)

#%% Copying files to CC folder
from shutil import copyfile

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