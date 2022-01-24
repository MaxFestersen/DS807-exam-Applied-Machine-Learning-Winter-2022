# -*- coding: utf-8 -*-
"""
Question 1

Created on Mon Jan 24 11:55:42 2022

@author: Max, Anders, Alexander
"""

#%% Importing libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from sklearn.metrics import accuracy_score
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV

# Set path to parrent location of current file
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
os.chdir("../")

#%% Question 1
print("Use non-deep learning to perform image classification according to the CC-D-Y modelling strategy. Specifically, you must:")

#%% Question 1.1
print("Discuss how the problem can be solved using support vector machines, random forests, and boosting (discuss each method separately).")
print("Discussion is done in report.")
# SVM er godt til at lave en baseline test. Det er nemt at sætte op, men vi får ikke andet end accuracy.
# Random forrest er meget nem at forstå logikken for indelingen. Så vi kan prøve at se hvilken af de 2 performer bedst.
# Boosting benytter mange modeller og anvender gennemsnit. Det kan være vi har en langsom learner, hvor det fungerer godt.

#%% Question 1.2
print("Use one of the methods above to solve the problem. A combination of two or all three of the methods may also be used, if you believe this is better (regardless of whether you use one or multiple methods, this must be motivated).")

#%% Read data
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

X_train, y_train = splitfolder_to_array(Categories=['0','1'], datadir='data/split/CC/train')
X_test, y_test = splitfolder_to_array(Categories=['0','1'], datadir='data/split/CC/test')
X_val, y_val = splitfolder_to_array(Categories=['0','1'], datadir='data/split/CC/val')
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape, X_val, y_val)

#%% Question 1.2 Problem solving: CC
#%% Question 1.2 Problem solving: CC SVM gridsearch
parameters = {'kernel':('linear', 'rbf'), 'C':[0.01, 0.05, 0.1, 0.5, 1]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(X_train, y_train)
sorted(clf.cv_results_.keys())
#%% Question 1.2 Problem solving: CC SVM loop
kernels = ["Linear", "rbf", "poly"]
Cs = [0.01, 0.05, 0.1, 0.5, 1]
results_C = []

for kernel in kernels:
    for C in Cs:
        svm_poly = svm.SVC(kernel=kernel, C=C)
        svm_poly.fit(X_train, y_train)
        y_val_hat = svm_poly.predict(X_val)
        accuracy = accuracy_score(y_val_hat, y_val)
        
        results_C.append([accuracy, kernel, C])

results_C = pd.DataFrame(results_C)
results_C.columns = ['Accuracy', 'Kernel', 'C']
print(results_C)

#%% Question 1.2 Problem solving: CC SVM Best model
best_k = results_C[results_C['Accuracy'] == results_C['Accuracy'].max()].iloc[0]['Kernel']
best_c = results_C[results_C['Accuracy'] == results_C['Accuracy'].max()].iloc[0]['C']

results_C[results_C['Accuracy'] == results_C['Accuracy'].max()]

svm_poly_best = svm.SVC(kernel=best_k, C = best_c)

# Use both training and validation data to fit it (np.concatenate "stacks" the array like rbind in R)
svm_poly_best.fit(np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val]))

# Predict on test data
y_val_hat_poly_best = svm_poly_best.predict(X_test)

# Obtain and check accuracy on test data
accuracy_poly_best = accuracy_score(y_val_hat_poly_best, y_test)
print(f'Optimized polynomial SVM achieved {round(accuracy_poly_best * 100, 1)}% accuracy on C.')

#%% Question 1.2 Problem solving: CC RF
#todo

#%% Question 1.2 Problem solving: CC B
#todo

#%% Question 1.2 Problem solving: D
#%% Question 1.2 Problem solving: D SVM
#todo

#%% Question 1.2 Problem solving: D RF
#todo

#%% Question 1.2 Problem solving: D B
#todo


#%% Question 1.2 Problem solving: Y
#%% Question 1.2 Problem solving: Y SVM
#todo

#%% Question 1.2 Problem solving: Y RF
#todo

#%% Question 1.2 Problem solving: Y B
#todo

#%% Question 1.2 Performance
print("Calculate and report the method’s performance on the training, validation, and test data.")

#%% Question 1.2 Performance: CC
#todo

#%% Question 1.2 Performance: D
#todo

#%% Question 1.2 Performance: Y
#todo

#%% Question 1.2 Performance-evaluation
print("Does the performance differ between the different sets? If yes, does this surprise you (explain why or why not)?")
print("Evaluation of performance is done in report.")
