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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from numpy import load
import joblib
from sklearn.metrics import accuracy_score, cohen_kappa_score, make_scorer
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

#%% Load numpy array from npy file - Anders
#%% Load numpy array from npy file - CC
X_train = load('data/X_train_CC.npy')
y_train = load('data/y_train_CC.npy')
X_test = load('data/X_test_CC.npy')
y_test = load('data/y_test_CC.npy')
X_val = load('data/X_val_CC.npy')
y_val = load('data/y_val_CC.npy')
#%% Load numpy array from npy file - D
X_train = load('data/X_train_D.npy')
y_train = load('data/y_train_D.npy')
X_test = load('data/X_test_D.npy')
y_test = load('data/y_test_D.npy')
X_val = load('data/X_val_D.npy')
y_val = load('data/y_val_D.npy')
#%% Load numpy array from npy file - Y
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
svm_CC = GridSearchCV(svc, 
                   parameters,
                   n_jobs=-1, # number of simultaneous jobs (-1 all cores)
                   scoring='balanced_accuracy')
svm_CC.fit(np.concatenate((X_train, X_val), axis=0), np.concatenate((y_train, y_val), axis=0))

results = pd.DataFrame(svm_CC.cv_results_)
print(results[results['mean_test_score'] == results['mean_test_score'].min()])

#%% Question 1.2 Problem solving: CC SVM gridsearch - Save results
joblib.dump(svm_CC, 'data/q12svmCC.pkl')

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

#%% Making tuning-grid for RF and GB - Anders

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
#
learning_rate = [0.15,0.1,0.05,0.01,0.005,0.001]

# Create grids
random_grid_RF = {'n_estimators': n_estimators,
                  'max_features': max_features,
                  'max_depth': max_depth,
                  'min_samples_split': min_samples_split,
                  'min_samples_leaf': min_samples_leaf,
                  'bootstrap': bootstrap,
                  'class_weight': class_weight}

random_grid_GB = {'max_depth': max_depth,
                  'min_samples_leaf': min_samples_leaf,
                  'learning_rate': learning_rate}

pprint(random_grid_RF)
pprint(random_grid_GB)
#%% Question 1.2 Problem solving: CC RF - Anders
#%% Standard model without tuning
# Initialize
rf = ensemble.RandomForestClassifier(random_state=(42))

# Fit
rf.fit(X_train, y_train)

# Predict
y_test_hat_std = rf.predict(X_test)
# accuracy and kappa score for evaluating performance
accuracy = accuracy_score(y_test, y_test_hat_std)
kappa = cohen_kappa_score(y_test, y_test_hat_std)
print(f'''RF with standard settings achieved {round(accuracy * 100, 1)}% accuracy and a kappa score of {round(kappa,2)}.''')
# confusion matrix
df_confusion = pd.crosstab(y_test, y_test_hat_std, rownames=['Actual'], colnames=['Predicted'],dropna=False)
plot_confusion_matrix(df_confusion)
#%% Save model 
joblib.dump(rf, 'data/q12rfCC_std.pkl')
#%% load model 
rf_CCc = joblib.load('data/q12rfCC_std.pkl')
#%%
kappa_scorer = make_scorer(cohen_kappa_score)
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf_CC = ensemble.RandomForestClassifier(random_state=42)
# Random search of parameters, using 3 fold cross validation, 
# search across 10 different combinations, use all available cores, and evaluate the performance with kappa
rf_CC = RandomizedSearchCV(rf_CC, random_grid_RF, n_iter = 10, cv = 3, verbose=2, random_state=42, n_jobs = -1, scoring='roc_auc')
# Fit the model
rf_CC.fit(np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val]))
# optimal parameter settings 
print(rf_CC.best_params_)
# predict
y_test_hat_rf_CC = rf_CC.predict(X_test)
# accuracy and kappa score for evaluating performance
accuracy = accuracy_score(y_test, y_test_hat_rf_CC)
kappa = cohen_kappa_score(y_test, y_test_hat_rf_CC)
print(f'''RF with tuned settings achieved {round(accuracy * 100, 1)}% accuracy and a kappa score of {round(kappa,2)}.''')
# confusion matrix
df_confusion = pd.crosstab(y_test, y_test_hat_rf_CC, rownames=['Actual'], colnames=['Predicted'],dropna=False)
plot_confusion_matrix(df_confusion)
#%% Save model 
joblib.dump(rf_CC, 'data/q12rfCC.pkl')
#%% load model 
rf_CC = joblib.load('data/q12rfCC.pkl')

#%%
from collections import Counter
import imblearn
from imblearn.ensemble import BalancedRandomForestClassifier
# Initialize
rf = imblearn.ensemble.BalancedRandomForestClassifier()

# Fit
rf.fit(X_train, y_train)

# Predict
y_test_hat = rf.predict(X_test)
# accuracy and kappa score for evaluating performance
accuracy = accuracy_score(y_test, y_test_hat)
kappa = cohen_kappa_score(y_test, y_test_hat)
print(f'''RF with tuned settings achieved {round(accuracy * 100, 1)}% accuracy and a kappa score of {round(kappa,2)}.''')
# confusion matrix
df_confusion = pd.crosstab(y_test, y_test_hat, rownames=['Actual'], colnames=['Predicted'],dropna=False)
plot_confusion_matrix(df_confusion)
#%% oversampling and under sampling 
# decision tree  on imbalanced dataset with SMOTE oversampling and random undersampling
from numpy import mean
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
#%%
# define pipeline
model = ensemble.RandomForestClassifier(random_state=42)
over = SMOTE(sampling_strategy=0.1)
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [('over', over), ('under', under), ('model', model)]
pipeline = Pipeline(steps=steps)
# evaluate pipeline

#rf_CCc = RandomizedSearchCV(model, random_grid_RF, n_iter = 10, cv = 3, verbose=2, random_state=42, n_jobs = -1, scoring=kappa_scorer)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(pipeline, X_train, y_train, scoring=[kappa_scorer, 'accuracy'], cv=cv, n_jobs=-1)
print('Mean kappa' mean(scores))
#%%
#from sklearn.model_selection import train_test_split
# transform the dataset
X, y = np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val])

over = SMOTE(sampling_strategy=0.2)
under = RandomUnderSampler(sampling_strategy=0.4)

steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)

X_train1, y_train1 = pipeline.fit_resample(X, y)
#X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2, random_state=42)

#%%
oversample = imblearn.over_sampling.SMOTE()
X, y = oversample.fit_resample(X_train, y_train)
counter = Counter(y)
print(counter)
#%%
rf_CC = ensemble.RandomForestClassifier(random_state=42)
# Random search of parameters, using 3 fold cross validation, 
# search across 10 different combinations, use all available cores, and evaluate the performance with kappa
rf_CC = RandomizedSearchCV(rf_CC, random_grid_RF, n_iter = 10, cv = 3, verbose=2, random_state=42, n_jobs = -1, scoring=kappa_scorer)
# Fit the model
rf_CC.fit(X_train1,y_train1)
# predict
y_test_hat_over = rf_CC.predict(X_test)
# accuracy and kappa score for evaluating performance
accuracy = accuracy_score(y_test, y_test_hat_over)
kappa = cohen_kappa_score(y_test, y_test_hat_over)
print(f'''RF with tuned settings achieved {round(accuracy * 100, 1)}% accuracy and a kappa score of {round(kappa,2)}.''')
# confusion matrix
df_confusion = pd.crosstab(y_test, y_test_hat_over, rownames=['Actual'], colnames=['Predicted'],dropna=False)
plot_confusion_matrix(df_confusion)

#%% - Anders
#df_confusion = pd.crosstab(y_test_CC, y_test_hat, rownames=['Actual'], colnames=['Predicted'],dropna=False)
def plot_confusion_matrix(df_confusion, title='Confusion matrix'):
    seaborn.heatmap(df_confusion, annot=True, fmt='d', cmap='Blues')
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
#%% Question 1.2 Problem solving: CC B - Anders
#todo

gbt = ensemble.HistGradientBoostingClassifier(random_state=42)

# Fit
gbt.fit(X_train_CC, y_train_CC)

# Predict
y_test_hat = gbt.predict(X_test_CC)
accuracy = accuracy_score(y_test_CC, y_test_hat)
kappa = cohen_kappa_score(y_test_CC, y_test_hat)
print(f'''Gradient boosted DTs with default settings achieved {round(accuracy * 100, 1)}% accuracy and a kappa score of {round(kappa,1)}.''')
df_confusion = pd.crosstab(y_test_CC, y_test_hat, rownames=['Actual'], colnames=['Predicted'],dropna=False)
plot_confusion_matrix(df_confusion)
#%% Question 1.2 Problem solving: D
#%% Question 1.2 Problem solving: D SVM
#%% Question 1.2 Problem solving: D SVM gridsearch - Scoring: balanced_accuracy
parameters = {'kernel':('rbf', 'linear', 'poly'), 'C':[1, 10, 100], 'gamma':['auto', 'scale'], 'decision_function_shape':['ovr', 'ovo']}
svc = svm.SVC()
svm_D = GridSearchCV(svc, 
                   parameters,
                   n_jobs=-1, # number of simultaneous jobs (-1 all cores)
                   scoring='balanced_accuracy')
svm_D.fit(np.concatenate((X_train, X_val), axis=0), np.concatenate((y_train, y_val), axis=0))

results = pd.DataFrame(svm_D.cv_results_)
print(results[results['mean_test_score'] == results['mean_test_score'].min()])

#%% Question 1.2 Problem solving: D SVM gridsearch - Save results
joblib.dump(svm_D, 'data/q12svmD_bacc.pkl')

#%% Question 1.2 Problem solving: D SVM gridsearch - Scoring: accuracy
parameters = {'kernel':('rbf', 'linear', 'poly'), 'C':[1, 10, 100], 'gamma':['auto', 'scale'], 'decision_function_shape':['ovr', 'ovo']}
svc = svm.SVC()
svm_D = GridSearchCV(svc, 
                   parameters,
                   n_jobs=-1, # number of simultaneous jobs (-1 all cores)
                   scoring='accuracy')
svm_D.fit(np.concatenate((X_train, X_val), axis=0), np.concatenate((y_train, y_val), axis=0))

results = pd.DataFrame(svm_D.cv_results_)
print(results[results['mean_test_score'] == results['mean_test_score'].min()])

#%% Question 1.2 Problem solving: D SVM gridsearch - Save results
joblib.dump(svm_D, 'data/q12svmD_acc.pkl')

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
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = ensemble.RandomForestClassifier(random_state=42)
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(rf, random_grid_RF, n_iter = 10, cv = 3, verbose=2, random_state=42, n_jobs = -1, scoring=kappa_scorer)
# Fit the random search model
rf_random.fit(X_train_D, y_train_D)

rf_random.best_params_
y_test_hat_std = rf_random.predict(X_test_D)
accuracy = accuracy_score(y_test_D, y_test_hat_std)
kappa = cohen_kappa_score(y_test_D, y_test_hat_std)
print(f'''RF with tuned settings achieved {round(accuracy * 100, 1)}% accuracy and a kappa score of {round(kappa,1)}.''')
df_confusion = pd.crosstab(y_test_D, y_test_hat_std, rownames=['Actual'], colnames=['Predicted'],dropna=False)
df_confusion = df_confusion.reindex(columns=[0,1,2,3,4,10], fill_value=0)
plot_confusion_matrix(df_confusion)
#%%
best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, test_features, test_labels)
#%% Question 1.2 Problem solving: D B
gbt_D = ensemble.HistGradientBoostingClassifier(random_state=(42), loss='categorical_crossentropy')

# Fit
gbt_D.fit(X_train_D, y_train_D)

# Predict
y_test_hat_D = gbt_D.predict(X_test_D)
accuracy = accuracy_score(y_test_D, y_test_hat_D)
print(f'''Gradient boosted DTs with default settings achieved {round(accuracy * 100, 1)}% accuracy.''')
df_confusion = pd.crosstab(y_test_D, y_test_hat, rownames=['Actual'], colnames=['Predicted'],dropna=False)
plot_confusion_matrix(df_confusion)
#%%
gb_D = ensemble.HistGradientBoostingClassifier(random_state=42,loss='categorical_crossentropy')
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
gb_random_D = RandomizedSearchCV(gb_D, random_grid_GB, n_iter = 10, cv = 3, random_state=42, n_jobs = -1)
# Fit the random search model
gb_random_D.fit(X_train_D, y_train_D)

gb_random.best_params_
y_test_hat_D_gb = gb_random.predict(X_test_D)
accuracy = accuracy_score(y_test_D, y_test_hat_D_gb)
#%% Question 1.2 Problem solving: Y
metrics.cohen_kappa_score
#%% Question 1.2 Problem solving: Y SVM
#%% Question 1.2 Problem solving: Y SVM gridsearch
parameters = {'kernel':('rbf', 'linear', 'poly'), 'C':[1, 10, 100], 'gamma':['auto', 'scale'], 'decision_function_shape':['ovr', 'ovo']}
svc = svm.SVC()
svm_Y = GridSearchCV(svc, 
                   parameters,
                   n_jobs=-1, # number of simultaneous jobs (-1 all cores)
                   scoring='balanced_accuracy')
svm_Y.fit(np.concatenate((X_train, X_val), axis=0), np.concatenate((y_train, y_val), axis=0))

results = pd.DataFrame(svm_Y.cv_results_)
print(results[results['mean_test_score'] == results['mean_test_score'].min()])

#%% Question 1.2 Problem solving: D SVM gridsearch - Save results
joblib.dump(svm_Y, 'data/q12svmY.pkl')


#%% Question 1.2 Problem solving: Y RF
#todo

#%% Question 1.2 Problem solving: Y B
#todo

#%% Question 1.2 Performance
print("Calculate and report the method’s performance on the training, validation, and test data.")

# Accuracy metric is bad for evaluating performance.

#%% Question 1.2 Performance: CC
#print(sorted(clf.cv_results_()))
svm_gridsearch_res = joblib.load("data/q12svmCC.pkl")
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
