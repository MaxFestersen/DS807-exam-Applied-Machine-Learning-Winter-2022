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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from numpy import save
from numpy import load
import joblib
from sklearn.metrics import accuracy_score
from sklearn import ensemble # ensemble instead of tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import seaborn
#from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint

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

X_train_D, y_train_D = splitfolder_to_array(Categories=['0','1','2','3','4','10'], datadir='data/split/D/train')
X_test_D, y_test_D = splitfolder_to_array(Categories=['0','1','2','3','4','10'], datadir='data/split/D/test')
X_val_D, y_val_D = splitfolder_to_array(Categories=['0','1','2','3','4','10'], datadir='data/split/D/val')
print(X_train_D.shape, X_test_D.shape, y_train_D.shape, y_test_D.shape, X_val_D.shape, y_val_D.shape)

X_train_Y, y_train_Y = splitfolder_to_array(Categories=['0','1','2','3','4','5','6','7','8','9','10'], datadir='data/split/Y/train')
X_test_Y, y_test_Y = splitfolder_to_array(Categories=['0','1','2','3','4','5','6','7','8','9','10'], datadir='data/split/Y/test')
X_val_Y, y_val_Y = splitfolder_to_array(Categories=['0','1','2','3','4','5','6','7','8','9','10'], datadir='data/split/Y/val')
print(X_train_Y.shape, X_test_Y.shape, y_train_Y.shape, y_test_Y.shape, X_val_Y.shape, y_val_Y.shape)


#%%
# save numpy array as npy file
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

#%%
# load numpy array from npy file
#%%
#CC
X_train = load('data/X_train_CC.npy')
y_train = load('data/y_train_CC.npy')
X_test = load('data/X_test_CC.npy')
y_test = load('data/y_test_CC.npy')
X_val = load('data/X_val_CC.npy')
y_val = load('data/y_val_CC.npy')
#%%
#D
X_train = load('data/X_train_D.npy')
y_train = load('data/y_train_D.npy')
X_test = load('data/X_test_D.npy')
y_test = load('data/y_test_D.npy')
X_val = load('data/X_val_D.npy')
y_val = load('data/y_val_D.npy')
#%%
#Y
X_train = load('data/X_train_Y.npy')
y_train = load('data/y_train_Y.npy')
X_test = load('data/X_test_Y.npy')
y_test = load('data/y_test_Y.npy')
X_val = load('data/X_val_Y.npy')
y_val = load('data/y_val_Y.npy')

#%% Scaling data

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

#%% Question 1.2 Problem solving: CC
#%% Question 1.2 Problem solving: CC SVM gridsearch
parameters = {'kernel':('rbf', 'linear', 'poly'), 'C':[1, 10, 100], 'gamma':['auto', 'scale']}
svc = svm.SVC()
clf = GridSearchCV(svc, 
                   parameters,
                   n_jobs=-1, # number of simultaneous jobs (-1 all cores)
                   scoring='balanced_accuracy')
clf.fit(np.concatenate((X_train, X_val), axis=0), np.concatenate((y_train, y_val), axis=0))

results = pd.DataFrame(clf.cv_results_)
print(results[results['mean_test_score'] == results['mean_test_score'].min()])

#%% Question 1.2 Problem solving: CC SVM gridsearch - Save results
joblib.dump(clf, 'data/q12svm.pkl')

#%% Question 1.2 Problem solving: CC SVM Best model
best_k = 'rbf'
best_c = 100

svm_poly_best = svm.SVC(kernel=best_k, C = best_c)

# Use both training and validation data to fit it (np.concatenate "stacks" the array like rbind in R)
svm_poly_best.fit(np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val]))

# Predict on test data
y_val_hat_poly_best = svm_poly_best.predict(X_test)

# Obtain and check accuracy on test data
accuracy_poly_best = accuracy_score(y_val_hat_poly_best, y_test)
print(f'Optimized polynomial SVM achieved {round(accuracy_poly_best * 100, 1)}% accuracy on C.')

#%% Question 1.2 Problem solving: CC RF
#%%
# Initialize
rf = ensemble.RandomForestClassifier(random_state=(42))

# Fit
rf.fit(X_train_CC, y_train_CC)

# Predict
y_test_hat_std = rf.predict(X_test_CC)
accuracy = accuracy_score(y_test_CC, y_test_hat_std)
print(f'''RF with default settings achieved {round(accuracy * 100, 1)}% accuracy.''')
df_confusion = pd.crosstab(y_test_CC, y_test_hat_std, rownames=['Actual'], colnames=['Predicted'],dropna=False)
plot_confusion_matrix(df_confusion)
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

#%%, cmap='spring'
#df_confusion = pd.crosstab(y_test_CC, y_test_hat, rownames=['Actual'], colnames=['Predicted'],dropna=False)
def plot_confusion_matrix(df_confusion, title='Confusion matrix'):
    seaborn.heatmap(df_confusion, annot=True, fmt='d')
    #plt.matshow(df_confusion, cmap=cmap) 
    #plt.colorbar()
    #tick_marks = np.arange(len(df_confusion.columns))
    #plt.yticks(tick_marks, df_confusion.columns)
    #plt.xticks(tick_marks, df_confusion.index)
    #plt.ylabel(df_confusion.index.name)
    #plt.xlabel(df_confusion.columns.name)
    #for i in range(len(df_confusion.columns)):
    #    for j in range(len(df_confusion.columns)):
    #        plt.text(j,i, str(df_confusion[i][j]))

#plot_confusion_matrix(df_confusion)
#%% Question 1.2 Problem solving: CC B
#todo
gbt = ensemble.HistGradientBoostingClassifier()

# Fit
gbt.fit(X_train_CC, y_train_CC)

# Predict
y_test_hat = gbt.predict(X_test_CC)
accuracy = accuracy_score(y_test_CC, y_test_hat)
print(f'''Gradient boosted DTs with default settings achieved {round(accuracy * 100, 1)}% accuracy.''')
df_confusion = pd.crosstab(y_test_CC, y_test_hat, rownames=['Actual'], colnames=['Predicted'],dropna=False)
plot_confusion_matrix(df_confusion)
#%% Question 1.2 Problem solving: D
#%% Question 1.2 Problem solving: D SVM
#todo

#%% Question 1.2 Problem solving: D RF
#todo
# Initialize
rf = ensemble.RandomForestClassifier(random_state=(42))

# Fit
rf.fit(X_train_D, y_train_D)

# Predict
y_test_hat_std = rf.predict(X_test_D)
accuracy = accuracy_score(y_test_D, y_test_hat_std)
print(f'''RF with default settings achieved {round(accuracy * 100, 1)}% accuracy.''')
print(confusion_matrix(y_test_D, y_test_hat_std))
df_confusion = pd.crosstab(y_test_D, y_test_hat_std, rownames=['Actual'], colnames=['Predicted'],dropna=False)
df_confusion = df_confusion.reindex(columns=[0,1,2,3,4,10], fill_value=0)
plot_confusion_matrix(df_confusion)
#%%

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'log2'] # auto = sqrt(n_features), log2 = log2(n_features)
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True]
#
class_weight=['balanced_subsample','balanced', None]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               'class_weight': class_weight}
pprint(random_grid)
#%%
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = ensemble.RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(rf, random_grid, n_iter = 10, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train_D, y_train_D)

rf_random.best_params_
y_test_hat_std = rf_random.predict(X_test_D)
accuracy = accuracy_score(y_test_D, y_test_hat_std)
print(f'''RF with tuned settings achieved {round(accuracy * 100, 1)}% accuracy.''')
df_confusion = pd.crosstab(y_test_D, y_test_hat_std, rownames=['Actual'], colnames=['Predicted'],dropna=False)
df_confusion = df_confusion.reindex(columns=[0,1,2,3,4,10], fill_value=0)
plot_confusion_matrix(df_confusion)
#%%
best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, test_features, test_labels)
#%% Question 1.2 Problem solving: D B
gbt_D = ensemble.HistGradientBoostingClassifier(random_state=(42))

# Fit
gbt.fit(X_train_D, y_train_D)

# Predict
y_test_hat_D = gbt_D.predict(X_test_D)
accuracy = accuracy_score(y_test_D, y_test_hat_D)
print(f'''Gradient boosted DTs with default settings achieved {round(accuracy * 100, 1)}% accuracy.''')
df_confusion = pd.crosstab(y_test_D, y_test_hat, rownames=['Actual'], colnames=['Predicted'],dropna=False)
plot_confusion_matrix(df_confusion)
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
svm_gridsearch_res = joblib.load("data/q12svm.pkl")
clf_predictions = svm_gridsearch_res.predict(X_test)

print(svm_gridsearch_res.best_estimator_)
print(svm_gridsearch_res.best_params_)
print(classification_report(y_test, clf_predictions))

#%% Question 1.2 Performance: D
#todo

#%% Question 1.2 Performance: Y
#todo

#%% Question 1.2 Performance-evaluation
print("Does the performance differ between the different sets? If yes, does this surprise you (explain why or why not)?")
print("Evaluation of performance is done in report.")
