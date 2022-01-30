# -*- coding: utf-8 -*-
"""
Question 1

Created on Mon Jan 24 11:55:42 2022

@author: Max, Anders, Alexander
"""

#%% Importing libraries
from collections import Counter
import imblearn
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
import joblib
import numpy as np
from numpy import load, mean
import matplotlib.pyplot as plt
import os
import pandas as pd
from pprint import pprint
import seaborn
from sklearn import datasets, ensemble, svm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, cohen_kappa_score, make_scorer, roc_auc_score, classification_report, ConfusionMatrixDisplay, confusion_matrix

# Set path to parrent location of current file - Max
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
os.chdir("../")

# Create kappa scorer - Anders
kappa_scorer = make_scorer(cohen_kappa_score)

#%% Initate pandas dataframe for comparing results - Max
if os.path.isfile('scores/nondeep.csv'):
    df_scores = pd.read_csv('scores/nondeep.csv')
else:
    df_scores = pd.DataFrame(columns=['Method_Category', 'Accuracy', 'Kappa', 'Roc'])
    os.makedirs('scores/', exist_ok=True)
    df_scores.to_csv("scores/nondeep.csv", index=False)

#%% Question 1.2

print("Use one of the methods above to solve the problem. A combination of two or all three of the methods may also be used, if you believe this is better (regardless of whether you use one or multiple methods, this must be motivated).")

#%% Load numpy array from npy file - CC - Anders

#%% Load numpy array from npy file - CC

X_train = load('data/X_train_CC.npy')
y_train = load('data/y_train_CC.npy')
X_test = load('data/X_test_CC.npy')
y_test = load('data/y_test_CC.npy')
X_val = load('data/X_val_CC.npy')
y_val = load('data/y_val_CC.npy')

#%% Load numpy array from npy file - D - Anders
X_train = load('data/X_train_D.npy')
y_train = load('data/y_train_D.npy')
X_test = load('data/X_test_D.npy')
y_test = load('data/y_test_D.npy')
X_val = load('data/X_val_D.npy')
y_val = load('data/y_val_D.npy')

#%% Load numpy array from npy file - Y - Anders
X_train = load('data/X_train_Y.npy')
y_train = load('data/y_train_Y.npy')
X_test = load('data/X_test_Y.npy')
y_test = load('data/y_test_Y.npy')
X_val = load('data/X_val_Y.npy')
y_val = load('data/y_val_Y.npy')

#%% Scaling data - Alexander
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

#%% Plot confusion matrix - Anders
def plot_confusion_matrix(df_confusion):
    x = df_confusion.reindex(columns=[x for x in range(len(Counter(y_test)))], fill_value=0)
    seaborn.heatmap(x, annot=True, fmt='d', cmap='Blues')
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

#%% Question 1.2 Problem solving: CC
#%% Question 1.2 Problem solving: CC SVM gridsearch - Max
parameters = {'kernel':('rbf', 'linear', 'poly'), 'C':[1, 10, 100], 'gamma':['auto', 'scale']}
svc = svm.SVC()
svm_CC = GridSearchCV(svc, 
                   parameters,
                   n_jobs=-1, # number of simultaneous jobs (-1 all cores)
                   scoring='balanced_accuracy')
svm_CC.fit(np.concatenate((X_train, X_val), axis=0), np.concatenate((y_train, y_val), axis=0))

results = pd.DataFrame(svm_CC.cv_results_)
print(results[results['mean_test_score'] == results['mean_test_score'].min()])

#%% Question 1.2 Problem solving: CC SVM gridsearch - Save results - Max
joblib.dump(svm_CC, 'data/q12svmCC.pkl')

#%% Question 1.2 Problem solving: CC SVM gridsearch - Predictions - Max
#predictions = svm_CC.predict(X_test)

#print(svm_CC.best_estimator_)
#print(svm_CC.best_params_)
#print(classification_report(y_test, predictions))
#print(svm_CC.cv_results_ )

# accuracy and kappa score for evaluating performance
#accuracy = accuracy_score(y_test, predictions)
#kappa = cohen_kappa_score(y_test, predictions)
#roc = roc_auc_score(y_test, predictions)
#print(f'SVM for CC achieved: {round(accuracy * 100, 1)}% accuracy, a kappa score of {round(kappa,2)} & roc score of {round(roc,2)}.')


#%% Question 1.2 Problem solving: CC RF - Anders
#%% Making tuning-grid for RF and GB - Anders
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'log2',2,4,6] # auto = sqrt(n_features), log2 = log2(n_features)
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4, 10, 20, 50]
# Method of selecting samples for training each tree
bootstrap = [True]
# for RF in the unbalanced CC
class_weight=['balanced_subsample','balanced', None]
# for boosting 
learning_rate = [0.15,0.1,0.05,0.01]
# Create grids
#Used for RF for the unbalanced CC
random_grid_RF_CC = {'n_estimators': n_estimators,
                     'max_features': max_features,
                     'max_depth': max_depth,
                     'min_samples_split': min_samples_split,
                     'min_samples_leaf': min_samples_leaf,
                     'bootstrap': bootstrap,
                     'class_weight': class_weight}
#Used for RF for the unbalanced CC with undersampling and D and Y
random_grid_RF = {'n_estimators': n_estimators,
                  'max_features': max_features,
                  'max_depth': max_depth,
                  'min_samples_split': min_samples_split,
                  'min_samples_leaf': min_samples_leaf,
                  'bootstrap': bootstrap}
# Used for boosting CC, D and Y
random_grid_GB = {'max_depth': max_depth,
                  'min_samples_leaf': min_samples_leaf,
                  'learning_rate': learning_rate}
pprint(random_grid_RF_CC)
pprint(random_grid_RF)
pprint(random_grid_GB)

#%% Standard model without tuning - Anders
# Initialize
rf_std_CC = ensemble.RandomForestClassifier(random_state=(42))
# Fit
rf_std_CC .fit(np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val]))
# Save model 
joblib.dump(rf_std_CC, 'data/q12rfCC_std.pkl')
#%% load model- Anders
rf_std_CC = joblib.load('data/q12rfCC_std.pkl')
# Predict
y_test_hat_std = rf_std_CC.predict(X_test)
# accuracy and kappa score for evaluating performance
accuracy = accuracy_score(y_test, y_test_hat_std)
kappa = cohen_kappa_score(y_test, y_test_hat_std)
roc_auc = roc_auc_score(y_test, y_test_hat_std)
print(f'''RF with standard settings achieved {round(accuracy * 100, 1)}% accuracy a kappa score of {round(kappa,2)} and roc_auc of {round(roc_auc,2)}.''')
# confusion matrix
df_confusion = pd.crosstab(y_test, y_test_hat_std, rownames=['Actual'], colnames=['Predicted'],dropna=False)
plot_confusion_matrix(df_confusion)
#%% optimize model based on kappa and AUC and save it.  - Anders
metrics = [kappa_scorer, 'roc_auc']
for i in metrics: 
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf_CC = ensemble.RandomForestClassifier(random_state=42)
    # Random search of parameters, using 3 fold cross validation, 
    # search across 10 (could be more) different combinations, use all available cores, and evaluate the performance with kappa and roc_auc
    rf_CC = RandomizedSearchCV(rf_CC, random_grid_RF_CC, n_iter = 10, cv = 3, verbose=2, random_state=42, n_jobs = -1, scoring=i) 
    # Fit the model
    rf_CC.fit(np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val]))
    joblib.dump(rf_CC, f'data/q12rfCC_{i}.pkl')
#%% load model ROC_AUC - Anders
rf_CC_ROC = joblib.load('data/q12rfCC_roc_auc.pkl')
print(rf_CC_ROC.best_params_,rf_CC_ROC.best_score_)

# predict
y_test_hat_rf_CC_ROC = rf_CC_ROC.predict(X_test)

# accuracy and kappa score for evaluating performance
accuracy = accuracy_score(y_test, y_test_hat_rf_CC_ROC)
kappa = cohen_kappa_score(y_test, y_test_hat_rf_CC_ROC)
roc_auc = roc_auc_score(y_test, y_test_hat_rf_CC_ROC)
print(f'''RF with tuned settings achieved {round(accuracy * 100, 1)}% accuracy a kappa score of {round(kappa,3)} and roc_auc of {round(roc_auc,3)}.''')

# confusion matrix
df_confusion = pd.crosstab(y_test, y_test_hat_rf_CC_ROC, rownames=['Actual'], colnames=['Predicted'],dropna=False)
plot_confusion_matrix(df_confusion)

#%% load model Kappa - Anders
rf_CC_kappa = joblib.load('data/q12rfCC_make_scorer(cohen_kappa_score).pkl')
print(rf_CC_kappa.best_params_, rf_CC_kappa.best_score_)
rf_CC_kappa.cv_results_['mean_test_score'] # mean validation score for each settings combination (10) 
# predict
y_test_hat_rf_CC_kappa = rf_CC_kappa.predict(X_test)

# accuracy and kappa score for evaluating performance
accuracy = accuracy_score(y_test, y_test_hat_rf_CC_kappa)
kappa = cohen_kappa_score(y_test, y_test_hat_rf_CC_kappa)
roc_auc = roc_auc_score(y_test, y_test_hat_rf_CC_kappa)
print(f'''RF with tuned settings achieved {round(accuracy * 100, 1)}% accuracy a kappa score of {round(kappa,3)} and roc_auc of {round(roc_auc,3)}.''')

# confusion matrix
df_confusion = pd.crosstab(y_test, y_test_hat_rf_CC_kappa, rownames=['Actual'], colnames=['Predicted'],dropna=False)
plot_confusion_matrix(df_confusion)

#%% ??? - Anders
# Initialize
for i in metrics: 
    rf_CC_bal = imblearn.ensemble.BalancedRandomForestClassifier(random_state=42)
    rf_CC_bal = RandomizedSearchCV(rf_CC_bal, random_grid_RF, n_iter = 10, cv = 3, verbose=2, random_state=42, n_jobs = -1, scoring=i)
    # Fit
    rf_CC_bal.fit(np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val]))
    joblib.dump(rf_CC_bal, f'data/q12rfCC_bal_{i}.pkl')
#%% - Anders
# Predict
rf_CC_bal_kappa = joblib.load('data/q12rfCC_bal_make_scorer(cohen_kappa_score).pkl')
print(rf_CC_bal_kappa.best_params_, rf_CC_bal_kappa.best_score_)
y_test_hat_CC_bal_kappa = rf_CC_bal_kappa.predict(X_test)
# accuracy and kappa score for evaluating performance
accuracy = accuracy_score(y_test, y_test_hat_CC_bal_kappa)
kappa = cohen_kappa_score(y_test, y_test_hat_CC_bal_kappa)
roc_auc = roc_auc_score(y_test, y_test_hat_CC_bal_kappa)
print(f'''RF with tuned settings achieved {round(accuracy * 100, 1)}% accuracy a kappa score of {round(kappa,3)} and roc_auc of {round(roc_auc,3)}.''')

# confusion matrix
df_confusion = pd.crosstab(y_test, y_test_hat_CC_bal_kappa, rownames=['Actual'], colnames=['Predicted'],dropna=False)
plot_confusion_matrix(df_confusion)

#%% - Anders
# Predict
rf_CC_bal_ROC = joblib.load('data/q12rfCC_bal_roc_auc.pkl')
print(rf_CC_bal_ROC.best_params_, rf_CC_bal_ROC.best_score_)
y_test_hat_CC_bal_ROC = rf_CC_bal_ROC.predict(X_test)
# accuracy and kappa score for evaluating performance
accuracy = accuracy_score(y_test, y_test_hat_CC_bal_ROC)
kappa = cohen_kappa_score(y_test, y_test_hat_CC_bal_ROC)
roc_auc = roc_auc_score(y_test, y_test_hat_CC_bal_ROC)
print(f'''RF with tuned settings achieved {round(accuracy * 100, 1)}% accuracy a kappa score of {round(kappa,3)} and roc_auc of {round(roc_auc,3)}.''')
# confusion matrix
df_confusion = pd.crosstab(y_test, y_test_hat_CC_bal_ROC, rownames=['Actual'], colnames=['Predicted'],dropna=False)
plot_confusion_matrix(df_confusion)

#%% Question 1.2 Problem solving: CC B - Anders
GB_CC_std = ensemble.HistGradientBoostingClassifier(random_state=42)
# Fit
GB_CC_std.fit(X_train, y_train)
# save model 
joblib.dump(GB_CC_std, 'data/q12GB_CC_std.pkl')
#%% load model  - Anders
GB_CC_std = joblib.load('data/q12GB_CC_std.pkl')
# Predict
y_test_hat_GB_CC_std = GB_CC_std.predict(X_test)
accuracy = accuracy_score(y_test, y_test_hat_GB_CC_std)
kappa = cohen_kappa_score(y_test, y_test_hat_GB_CC_std)
roc_auc = roc_auc_score(y_test, y_test_hat_GB_CC_std)
print(f'''RF with tuned settings achieved {round(accuracy * 100, 1)}% accuracy a kappa score of {round(kappa,3)} and roc_auc of {round(roc_auc,3)}.''')
df_confusion = pd.crosstab(y_test, y_test_hat_GB_CC_std, rownames=['Actual'], colnames=['Predicted'],dropna=False)
plot_confusion_matrix(df_confusion)
#%% tune model - Anders
#parameters to tune  
GB_CC = ensemble.HistGradientBoostingClassifier(random_state=42)
# Random search of parameters, using 3 fold cross validation, 
# search across 10 different combinations, use all available cores, and evaluate the performance with kappa
GB_CC = RandomizedSearchCV(GB_CC, random_grid_GB, n_iter = 10, cv = 3, verbose=2, random_state=42, n_jobs = -1, scoring=kappa_scorer)
# Fit the model
GB_CC.fit(np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val]))
joblib.dump(GB_CC, 'data/q12GB_CC.pkl')
#%% - Anders
GB_CC = joblib.load('data/q12GB_CC.pkl')

y_test_hat_GB_CC = GB_CC.predict(X_test)
accuracy = accuracy_score(y_test, y_test_hat_GB_CC)
kappa = cohen_kappa_score(y_test, y_test_hat_GB_CC)
roc_auc = roc_auc_score(y_test, y_test_hat_GB_CC)
print(f'''RF with tuned settings achieved {round(accuracy * 100, 1)}% accuracy a kappa score of {round(kappa,3)} and roc_auc of {round(roc_auc,3)}.''')
df_confusion = pd.crosstab(y_test, y_test_hat_GB_CC, rownames=['Actual'], colnames=['Predicted'],dropna=False)
plot_confusion_matrix(df_confusion)
#%% ??? - Anders
for i in metrics:
    GB_CC = ensemble.HistGradientBoostingClassifier(random_state=42)
    # Random search of parameters, using 3 fold cross validation, 
    # search across 10 different combinations, use all available cores, and evaluate the performance with kappa
    GB_CC = RandomizedSearchCV(GB_CC, random_grid_GB, n_iter = 10, cv = 3, verbose=2, random_state=42, n_jobs = -1, scoring=i)
    # Fit the model
    GB_CC.fit(np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val]))
    joblib.dump(GB_CC, f'data/GB_CC_{i}.pkl')
#%% load model kappa - Anders
GB_CC_kappa = joblib.load('data/GB_CC_make_scorer(cohen_kappa_score).pkl')
# Predict
print(GB_CC_kappa.best_params_, GB_CC_kappa.best_score_)
y_test_hat_GB_CC_std_kappa = GB_CC_kappa.predict(X_test)
accuracy = accuracy_score(y_test, y_test_hat_GB_CC_std_kappa)
kappa = cohen_kappa_score(y_test, y_test_hat_GB_CC_std_kappa)
roc_auc = roc_auc_score(y_test, y_test_hat_GB_CC_std_kappa)
print(f'''RF with tuned settings achieved {round(accuracy * 100, 1)}% accuracy a kappa score of {round(kappa,3)} and roc_auc of {round(roc_auc,3)}.''')
df_confusion = pd.crosstab(y_test, y_test_hat_GB_CC_std_kappa, rownames=['Actual'], colnames=['Predicted'],dropna=False)
plot_confusion_matrix(df_confusion)
#%% - Anders
GB_CC_ROC = joblib.load('data/GB_CC_roc_auc.pkl')
# Predict
print(GB_CC_ROC.best_params_, GB_CC_ROC.best_score_)
y_test_hat_GB_CC_std_ROC = GB_CC_ROC.predict(X_test)
accuracy = accuracy_score(y_test, y_test_hat_GB_CC_std_ROC)
kappa = cohen_kappa_score(y_test, y_test_hat_GB_CC_std_ROC)
roc_auc = roc_auc_score(y_test, y_test_hat_GB_CC_std_ROC)
print(f'''RF with tuned settings achieved {round(accuracy * 100, 1)}% accuracy a kappa score of {round(kappa,3)} and roc_auc of {round(roc_auc,3)}.''')
df_confusion = pd.crosstab(y_test, y_test_hat_GB_CC_std_ROC, rownames=['Actual'], colnames=['Predicted'],dropna=False)
plot_confusion_matrix(df_confusion)
#%% - Anders
gbt = imblearn.ensemble.EasyEnsembleClassifier(random_state=42)
gbt.fit(np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val]))
joblib.dump(gbt, 'data/GB_CC_Eensemble.pkl')
#%% - Anders
gbt = joblib.load('data/GB_CC_Eensemble.pkl')
# Predict
y_test_hat = gbt.predict(X_test)
accuracy = accuracy_score(y_test, y_test_hat)
kappa = cohen_kappa_score(y_test, y_test_hat)
roc_auc = roc_auc_score(y_test, y_test_hat)
print(f'''RF with tuned settings achieved {round(accuracy * 100, 1)}% accuracy a kappa score of {round(kappa,3)} and roc_auc of {round(roc_auc,3)}.''')
df_confusion = pd.crosstab(y_test, y_test_hat, rownames=['Actual'], colnames=['Predicted'],dropna=False)
plot_confusion_matrix(df_confusion)

#%% Question 1.2 Problem solving: D
#%% Question 1.2 Problem solving: D SVM - Max
#%% Question 1.2 Problem solving: D SVM gridsearch - Scoring: balanced_accuracy - Max
parameters = {'kernel':['rbf'], 'C':[10, 100], 'gamma':['auto', 'scale'], 'decision_function_shape':['ovr']}
svc = svm.SVC(probability=True)
svm_D = GridSearchCV(svc,
                     parameters,
                     n_jobs=-1, # number of simultaneous jobs (-1 all cores)
                     scoring='balanced_accuracy')
svm_D.fit(np.concatenate((X_train, X_val), axis=0), np.concatenate((y_train, y_val), axis=0))

results = pd.DataFrame(svm_D.cv_results_)
print(results[results['mean_test_score'] == results['mean_test_score'].min()])

#%% Question 1.2 Problem solving: D SVM gridsearch - Save results - Max
joblib.dump(svm_D, 'data/q12svmD.pkl')

#%% Question 1.2 Problem solving: D SVM gridsearch - Prediction - Max
# predictions = svm_D.predict(X_test)
# proba_pred = svm_D.predict_proba(X_test)
#print(svm_D.best_estimator_)
#print(svm_D.best_params_)
#print(svm_D(y_test, predictions))

#%% Question 1.2 Problem solving: D SVM Best model - Max
svm_best = svm.SVC(kernel='rbf', C = 100, gamma = 'auto', decision_function_shape = "ovr", probability = True)
svm_best.fit(X_train, y_train)


#%% Question 1.2 Problem solving: D RF - Anders
# Initialize
rf_D_std = ensemble.RandomForestClassifier(random_state=(42))

# Fit
rf_D_std.fit(np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val]))
joblib.dump(rf_D_std, 'data/q12rf_D_std.pkl')
#%%- Anders
rf_D_std = joblib.load('data/q12rf_D_std.pkl')
# Predict
y_test_hat_D_std = rf_D_std.predict(X_test)
accuracy = accuracy_score(y_test, y_test_hat_D_std)
kappa = cohen_kappa_score(y_test, y_test_hat_D_std)
#roc_auc = roc_auc_score(y_test, y_test_hat_D_std)
print(f'''RF with tuned settings achieved {round(accuracy * 100, 1)}% accuracy and a kappa score of {round(kappa,3)}.''')
df_confusion = pd.crosstab(y_test, y_test_hat_D_std, rownames=['Actual'], colnames=['Predicted'],dropna=False)
#df_confusion = df_confusion.reindex(columns=[0,1,2,3,4,5], fill_value=0)
plot_confusion_matrix(df_confusion)
#%%- Anders
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = ensemble.RandomForestClassifier(random_state=42)
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(rf, random_grid_RF, n_iter = 10, cv = 3, verbose=2, random_state=42, n_jobs = -1, scoring='accuracy')
# Fit the random search model
rf_random.fit(np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val]))
joblib.dump(rf_random, 'data/q12rf_D.pkl')
#%%- Anders
rf_D_tuned = joblib.load('data/q12rf_D.pkl')
print(rf_D_tuned.best_params_, rf_D_tuned.best_score_)
y_test_hat_D_tuned = rf_D_tuned.predict(X_test)
accuracy = accuracy_score(y_test, y_test_hat_D_tuned)
kappa = cohen_kappa_score(y_test, y_test_hat_D_tuned)
print(f'''RF with tuned settings achieved {round(accuracy * 100, 1)}% accuracy and a kappa score of {round(kappa,1)}.''')
df_confusion = pd.crosstab(y_test, y_test_hat_D_tuned, rownames=['Actual'], colnames=['Predicted'],dropna=False)
df_confusion = df_confusion.reindex(columns=[0,1,2,3,4,10], fill_value=0)
plot_confusion_matrix(df_confusion)
#%% Question 1.2 Problem solving: D B - Anders
gbt_D = ensemble.HistGradientBoostingClassifier(random_state=(42))
# Fit
gbt_D.fit(np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val]))
joblib.dump(gbt_D, 'data/q12gbt_D.pkl')
#%%- Anders
gbt_D_std = joblib.load('data/q12gbt_D.pkl')
# Predict
y_test_hat_D = gbt_D_std.predict(X_test)
accuracy = accuracy_score(y_test, y_test_hat_D)
kappa = cohen_kappa_score(y_test, y_test_hat_D)
print(f'''RF with tuned settings achieved {round(accuracy * 100, 1)}% accuracy and a kappa score of {round(kappa,1)}.''')
df_confusion = pd.crosstab(y_test, y_test_hat_D, rownames=['Actual'], colnames=['Predicted'],dropna=False)
plot_confusion_matrix(df_confusion)
#%%- Anders
gb_D = ensemble.HistGradientBoostingClassifier(random_state=42)
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
gb_D = RandomizedSearchCV(gb_D, random_grid_GB, n_iter = 10, cv = 3, random_state=42, n_jobs = -1)
# Fit the random search model
gb_D.fit(np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val]))
joblib.dump(gb_D, 'data/q12gb_D.pkl')
#%%- Anders
gb_D_tuned = joblib.load('data/q12gb_D.pkl')
print(gb_D_tuned.best_params_, gb_D_tuned.best_score_)
gb_D_tuned.best_params_
y_test_hat_D_Tuned = gb_D_tuned.predict(X_test)
accuracy = accuracy_score(y_test, y_test_hat_D_Tuned)
kappa = cohen_kappa_score(y_test, y_test_hat_D_Tuned)
print(f'''RF with tuned settings achieved {round(accuracy * 100, 1)}% accuracy and a kappa score of {round(kappa,1)}.''')
df_confusion = pd.crosstab(y_test, y_test_hat_D_Tuned, rownames=['Actual'], colnames=['Predicted'],dropna=False)
plot_confusion_matrix(df_confusion)
#%% Question 1.2 Problem solving: Y
metrics.cohen_kappa_score
#%% Question 1.2 Problem solving: Y SVM - Max
#%% Question 1.2 Problem solving: Y SVM gridsearch - Max
parameters = {'kernel':['rbf'], 'C':[1, 10, 100], 'gamma':['auto', 'scale'], 'decision_function_shape':['ovr']}
svc = svm.SVC(probability=True)
svm_Y = GridSearchCV(svc, 
                   parameters,
                   n_jobs=-1, # number of simultaneous jobs (-1 all cores)
                   scoring='accuracy')
svm_Y.fit(np.concatenate((X_train, X_val), axis=0), np.concatenate((y_train, y_val), axis=0))

results = pd.DataFrame(svm_Y.cv_results_)
print(results[results['mean_test_score'] == results['mean_test_score'].min()])

#%% Question 1.2 Problem solving: Y SVM gridsearch - Save results - Max
joblib.dump(svm_Y, 'data/q12svmY.pkl')

#%% Question 1.2 Problem solving: Y SVM gridsearch - Predictions - Max
# predictions = svm_Y.predict(X_test)
# proba_pred = svm_Y.predict_proba(X_test)

#print(svm_Y.best_estimator_)
#print(svm_Y.best_params_)
#print(svm_Y(y_test, predictions))

#%% Question 1.2 Problem solving: Y SVM Best model - Max
svm_best = svm.SVC(kernel='rbf', C = 10, gamma = 'auto', decision_function_shape = "ovr", probability = True)
svm_best.fit(X_train, y_train)

#%% Question 1.2 Problem solving: Y RF - Anders 
# Initialize
rf_Y_std = ensemble.RandomForestClassifier(random_state=(42))

# Fit
rf_Y_std.fit(np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val]))
joblib.dump(rf_Y_std, 'data/q12rf_Y_std.pkl')
#%%- Anders
rf_Y_std = joblib.load('data/q12rf_Y_std.pkl')
# Predict
y_test_hat_T_std = rf_Y_std.predict(X_test)
accuracy = accuracy_score(y_test, y_test_hat_T_std)
kappa = cohen_kappa_score(y_test, y_test_hat_T_std)
#roc_auc = roc_auc_score(y_test, y_test_hat_D_std)
print(f'''RF with tuned settings achieved {round(accuracy * 100, 1)}% accuracy and a kappa score of {round(kappa,3)}.''')
df_confusion = pd.crosstab(y_test, y_test_hat_T_std, rownames=['Actual'], colnames=['Predicted'],dropna=False)
#df_confusion = df_confusion.reindex(columns=[0,1,2,3,4,5,6,7,8,9,10], fill_value=0)
plot_confusion_matrix(df_confusion)
#%%- Anders
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf_Y = ensemble.RandomForestClassifier(random_state=42)
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_Y = RandomizedSearchCV(rf_Y, random_grid_RF, n_iter = 10, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_Y.fit(np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val]))
joblib.dump(rf_Y, 'data/q12rf_Y.pkl')
#%%- Anders
rf_Y = joblib.load('data/q12rf_Y.pkl')
print(rf_Y.best_params_)
y_test_hat_Y = rf_Y.predict(X_test)
accuracy = accuracy_score(y_test, y_test_hat_Y)
kappa = cohen_kappa_score(y_test, y_test_hat_Y)
print(f'''RF with tuned settings achieved {round(accuracy * 100, 1)}% accuracy and a kappa score of {round(kappa,1)}.''')
df_confusion = pd.crosstab(y_test, y_test_hat_Y, rownames=['Actual'], colnames=['Predicted'],dropna=False)
#df_confusion = df_confusion.reindex(columns=[0,1,2,3,4,5,6,7,8,9,10], fill_value=0)
plot_confusion_matrix(df_confusion)
#%% Question 1.2 Problem solving: Y B - Anders
gbt_Y = ensemble.HistGradientBoostingClassifier(random_state=(42)) # loss auto
# Fit
gbt_Y.fit(np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val]))
joblib.dump(gbt_Y, 'data/q12gbt_Y.pkl')
#%%- Anders
gbt_Y_std = joblib.load('data/q12gbt_Y.pkl')
# Predict
y_test_hat_Y = gbt_Y_std.predict(X_test)
accuracy = accuracy_score(y_test, y_test_hat_Y)
kappa = cohen_kappa_score(y_test, y_test_hat_Y)
print(f'''RF with tuned settings achieved {round(accuracy * 100, 1)}% accuracy and a kappa score of {round(kappa,1)}.''')
df_confusion = pd.crosstab(y_test, y_test_hat_Y, rownames=['Actual'], colnames=['Predicted'],dropna=False)
#df_confusion = df_confusion.reindex(columns=[0,1,2,3,4,5,6,7,8,9,10], fill_value=0)
plot_confusion_matrix(df_confusion)
#%%- Anders
gb_Y = ensemble.HistGradientBoostingClassifier(random_state=42) # loss = 'auto' 
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
gb_Y = RandomizedSearchCV(gb_Y, random_grid_GB, n_iter = 10, cv = 3, random_state=42, n_jobs = -1)
# Fit the random search model
gb_Y.fit(np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val]))
joblib.dump(gb_Y, 'data/q12gb_Y.pkl')
#%%- Anders
gb_Y_tuned = joblib.load('data/q12gb_Y.pkl')
print(gb_Y_tuned.best_params_)
y_test_hat_Y_Tuned = gb_Y_tuned.predict(X_test)
accuracy = accuracy_score(y_test, y_test_hat_Y_Tuned)
kappa = cohen_kappa_score(y_test, y_test_hat_Y_Tuned)
print(f'''RF with tuned settings achieved {round(accuracy * 100, 1)}% accuracy and a kappa score of {round(kappa,1)}.''')
df_confusion = pd.crosstab(y_test, y_test_hat_Y_Tuned, rownames=['Actual'], colnames=['Predicted'],dropna=False)
plot_confusion_matrix(df_confusion)

