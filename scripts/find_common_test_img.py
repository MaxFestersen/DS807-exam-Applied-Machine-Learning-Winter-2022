#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 17:18:04 2022

@author: adernild
"""
import os
from shutil import copyfile

path_CC = 'data/split/CC/test'
path_D = 'data/split/D/test'
path_Y = 'data/split/Y/test'

test_img = []

for i in os.listdir(path_CC):
    for j in os.listdir(os.path.join(path_CC, i)):
        test_img.append(j)

for i in os.listdir(path_D):
    for j in os.listdir(os.path.join(path_D, i)):
        test_img.append(j)

for i in os.listdir(path_Y):
    for j in os.listdir(os.path.join(path_Y, i)):
        test_img.append(j)
        
test = set([x for x in test_img if test_img.count(x) > 2]) # images in all three test sets

#%% move test images present in all three test sets to data/test
for i in test:
    src = os.path.join('data/DIDA_2/', i)
    dst = os.path.join('data/test/', i)
    if not os.path.exists('data/test'):
        os.makedirs('data/test')
    copyfile(src, dst)
