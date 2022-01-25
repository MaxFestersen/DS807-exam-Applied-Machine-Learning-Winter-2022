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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

X_train_CC, y_train_CC = splitfolder_to_array(Categories=['0','1'], datadir='data/split/CC/train')
X_test_CC, y_test_CC = splitfolder_to_array(Categories=['0','1'], datadir='data/split/CC/test')
X_val_CC, y_val_CC = splitfolder_to_array(Categories=['0','1'], datadir='data/split/CC/val')
print(X_train_CC.shape, X_test_CC.shape, y_train_CC.shape, y_test_CC.shape, X_val_CC.shape, y_val_CC.shape)

X_train_D, y_train_D = splitfolder_to_array(Categories=['0','1','2','3','4','5','6','7','8','9','10'], datadir='data/split/D/train')
X_test_D, y_test_D = splitfolder_to_array(Categories=['0','1','2','3','4','5','6','7','8','9','10'], datadir='data/split/D/test')
X_val_D, y_val_D = splitfolder_to_array(Categories=['0','1','2','3','4','5','6','7','8','9','10'], datadir='data/split/D/val')
print(X_train_D.shape, X_test_D.shape, y_train_D.shape, y_test_D.shape, X_val_D.shape, y_val_D.shape)

X_train_Y, y_train_Y = splitfolder_to_array(Categories=['0','1','2','3','4','5','6','7','8','9','10'], datadir='data/split/Y/train')
X_test_Y, y_test_Y = splitfolder_to_array(Categories=['0','1','2','3','4','5','6','7','8','9','10'], datadir='data/split/Y/test')
X_val_Y, y_val_Y = splitfolder_to_array(Categories=['0','1','2','3','4','5','6','7','8','9','10'], datadir='data/split/Y/val')
print(X_train_Y.shape, X_test_Y.shape, y_train_Y.shape, y_test_Y.shape, X_val_Y.shape, y_val_Y.shape)
#%%
# save numpy array as npy file
#from numpy import asarray
from numpy import save
l1 = ['X_train_', 'y_train_', 'X_test_', 'y_test_', 'X_val_', 'y_val_']
l2 = ['CC','D','Y']
for i in l1:
    for j in l2:
        a=f"data/{i}{j}"
        b=f'{i}{j}'
        save(a, b)
#%%        
# save to npy file
save('data/X_train_CC.npy', X_train_CC)
save('data/y_train_CC.npy', y_train_CC)
save('data/X_test_CC.npy', X_test_CC)
save('data/y_test_CC.npy', y_test_CC)
save('data/X_val_CC.npy', X_val_CC)
save('data/y_val_CC.npy', y_val_CC)
#%%
# load numpy array from npy file
from numpy import load
# load array
data = load('data/data.npy')
# print the array
print(data)
#%% Question 1.2 Problem solving: CC
#%% Question 1.2 Problem solving: CC SVM gridsearch
parameters = {'kernel':('linear', 'rbf'), 'C':[1], 'gamma':[1, 0.01, 0.0001]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(np.concatenate((X_train_CC, X_val_CC), axis=0), np.concatenate((y_train_CC, y_val_CC), axis=0))
sorted(clf.cv_results_.keys())


#%% Question 1.2 Problem solving: CC SVM loop
kernels = ["linear", "rbf", "poly"]
Cs = [0.01]
gammas = [1, 0.01, 0.0001]
results_C = []

for kernel in kernels:
    for C in Cs:
        for gamma in gammas:
            svm= svm.SVC(kernel=kernel, C=C, gamma=gamma)
            svm.fit(X_train_CC, y_train_CC)
            y_val_hat = svm.predict(X_val_CC)
            accuracy = accuracy_score(y_val_hat, y_val_CC)
            
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
svm_poly_best.fit(np.concatenate([X_train_CC, X_val_CC]), np.concatenate([y_train_CC, y_val_CC]))

# Predict on test data
y_val_hat_poly_best = svm_poly_best.predict(X_test_CC)

# Obtain and check accuracy on test data
accuracy_poly_best = accuracy_score(y_val_hat_poly_best, y_test_CC)
print(f'Optimized polynomial SVM achieved {round(accuracy_poly_best * 100, 1)}% accuracy on C.')

#%% Question 1.2 Problem solving: CC RF
from sklearn.metrics import accuracy_score
from sklearn import ensemble # ensemble instead of tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import seaborn
#%%
# Initialize
rf = ensemble.RandomForestClassifier(random_state=(42))

# Fit
rf.fit(X_train_CC, y_train_CC)

# Predict
y_test_hat_std = rf.predict(X_test_CC)
accuracy = accuracy_score(y_test_CC, y_test_hat_std)
print(f'''RF with default settings achieved {round(accuracy * 100, 1)}% accuracy.''')
#%%

# Initialize
rf = ensemble.RandomForestClassifier(class_weight='balanced', random_state=(42))

# Fit
rf.fit(X_train_CC, y_train_CC)

# Predict
y_test_hat_bal = rf.predict(X_test_CC)
accuracy = accuracy_score(y_test_CC, y_test_hat_bal)
print(f'''RF with default settings achieved {round(accuracy * 100, 1)}% accuracy.''')

#%%
# Initialize
rf = ensemble.RandomForestClassifier(class_weight='balanced_subsample', random_state=(42))

# Fit
rf.fit(X_train_CC, y_train_CC)

# Predict
y_test_hat_sub = rf.predict(X_test_CC)
accuracy = accuracy_score(y_test_CC, y_test_hat_sub)
print(f'''RF with default settings achieved {round(accuracy * 100, 1)}% accuracy.''')
#%%
import imbalanced_learn as imblearn
from imblearn.ensemble import BalancedRandomForestClassifier
# Initialize
rf = ensemble.BalancedRandomForestClassifier()

# Fit
rf.fit(X_train_CC, y_train_CC)

# Predict
y_test_hat = rf.predict(X_test_CC)
accuracy = accuracy_score(y_test_CC, y_test_hat)
print(f'''RF with default settings achieved {round(accuracy * 100, 1)}% accuracy.''')

#%%

def make_confusion_matrix(true, pred, class_name1, class_name2):
    cm = confusion_matrix(true, pred)
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = [class_name1,class_name2]
    plt.title(f'{class_name1} or Not Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames) 
    plt.yticks(tick_marks, classNames)
    s = [['TN','FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
    plt.show()
    
make_confusion_matrix(y_test_CC, y_test_hat_sub,'CC_18', 'CC_other')
#%% Question 1.2 Problem solving: CC B
#todo
gbt = ensemble.HistGradientBoostingClassifier()

# Fit
gbt.fit(X_train_CC, y_train_CC)
#%%
# Predict
y_test_hat = gbt.predict(X_test_CC)
accuracy = accuracy_score(y_test_CC, y_test_hat)
print(f'''Gradient boosted DTs with default settings achieved {round(accuracy * 100, 1)}% accuracy.''')
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

# Accuracy metric is bad for evaluating performance.

#%% Question 1.2 Performance: CC
#print(sorted(clf.cv_results_()))
clf_predictions = clf.predict(X_test_CC)

print(clf.best_estimator_)
print(clf.best_params_)
print(classification_report(y_test_CC, clf_predictions))

#%% Question 1.2 Performance: D
#todo

#%% Question 1.2 Performance: Y
#todo

#%% Question 1.2 Performance-evaluation
print("Does the performance differ between the different sets? If yes, does this surprise you (explain why or why not)?")
print("Evaluation of performance is done in report.")
