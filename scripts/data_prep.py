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

# Set path to parrent location of current file
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
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
