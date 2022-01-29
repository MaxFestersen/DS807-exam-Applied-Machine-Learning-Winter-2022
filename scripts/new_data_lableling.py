# -*- coding: utf-8 -*-
"""
Moving random subset of new images
"""

#%% Importing libraries
import os
import random
import shutil

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
