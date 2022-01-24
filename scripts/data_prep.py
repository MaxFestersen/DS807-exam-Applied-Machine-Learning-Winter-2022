# -*- coding: utf-8 -*-
"""
Data preparation
"""

#%% Importing libraries
import pandas as pd
import numpy as np
import os
os.chdir("../")

#%% Importing data and encoding new colums

df = pd.read_csv("data/DIDA_12000_String_Digit_Labels.csv", header=None)

df['path'] = "DIDA_1/" + df[0].astype(str) + ".jpg"
df['CC'] = np.select([(df[1]<1900) & (df[1]>1799)], [0], default=1)
df['D'] = np.select([(df[1].astype(str).str.len()>1) & (df[1].astype(str).str.len()<5)], [df[1].astype(str).str[-2]], default=10)
df['Y'] = np.select([(df[1].astype(str).str.len()>1) & (df[1].astype(str).str.len()<5)], [df[1].astype(str).str[-1]], default=10)

