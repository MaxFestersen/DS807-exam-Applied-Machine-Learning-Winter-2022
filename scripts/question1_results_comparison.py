# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 22:08:08 2022

@author: Max

Results for Q1.2
"""
from collections import Counter
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

# --------------- Question 1.2 - Predictions - Max ---------------------------
#%% Initate pandas dataframe for comparing results
if os.path.isfile('scores/nondeep.csv'):
    df_scores = pd.read_csv('scores/nondeep.csv')
else:
    df_scores = pd.DataFrame(columns=['Method_Category', 'Accuracy', 'Kappa', 'Roc'])
    os.makedirs('scores/', exist_ok=True)
    df_scores.to_csv("scores/nondeep.csv", index=False)

#%% Result handler
def add_model_score(name, model, df_scores, svm_multi_class): # ROC requires multiclass parameters.
    X_test_pred = model.predict(X_test)
    X_train_pred = model.predict(X_train)

    # Obtain and check accuracy on test data
    X_test_acc = accuracy_score(X_test_pred, y_test)
    # X_test_acc = balanced_accuracy_score(X_test_pred, y_test, sample_weight = X_train_weights_by_class)
    X_test_kappa = cohen_kappa_score(y_test, X_test_pred)
    if svm_multi_class == 1:
        proba_pred = model.predict_proba(X_test)
        X_test_roc = roc_auc_score(y_test, proba_pred, multi_class="ovr")
    elif  svm_multi_class == 2:
        X_test_roc = roc_auc_score(y_test, X_test_pred)
    else:
        X_test_roc = np.nan
    print(f'{name} achieved on test-set: {round(X_test_acc * 100, 1)}% accuracy, a kappa score of {round(X_test_kappa,2)} & roc score of {round(X_test_roc,2)}.')
    if df_scores.loc[df_scores['Method_Category'] == f"{name} test"].empty:
        print(f"Adding {name} test-set.")
        new_row = {'Method_Category': f"{name} test", 'Accuracy': X_test_acc, 'Kappa': X_test_kappa, 'Roc': X_test_roc}
        df_scores = df_scores.append(new_row, ignore_index = True)
        df_scores.to_csv("scores/nondeep.csv", index=False)
    else:
        print(f"Updating {name} test-set.")
        df_scores.loc[df_scores['Method_Category'] == f"{name} test"] = f"{name} test", X_test_acc, X_test_kappa, X_test_roc
        df_scores.to_csv("scores/nondeep.csv", index=False)
    
    # Obtain and check accuracy on validation data
    X_val_acc = model.best_score_
    print(f'{name} achieved a mean accuracy of {round(X_val_acc * 100, 1)}% on it\'s validations.')
    if df_scores.loc[df_scores['Method_Category'] == f"{name} val"].empty:
        print(f"Adding {name} validation-set.")
        new_row = {'Method_Category': f"{name} val", 'Accuracy': X_val_acc, 'Kappa': None, 'Roc': None}
        df_scores = df_scores.append(new_row, ignore_index = True)
        df_scores.to_csv("scores/nondeep.csv", index=False)
    else:
        print(f"Updating {name} test-set.")
        df_scores.loc[df_scores['Method_Category'] == f"{name} val"] = f"{name} val", X_val_acc, None, None
        df_scores.to_csv("scores/nondeep.csv", index=False)
    
    # Obtain and check accuracy on training data
    X_train_acc = accuracy_score(X_train_pred, y_train)
    X_train_kappa = cohen_kappa_score(y_train, X_train_pred)
    if svm_multi_class == 1:
        proba_pred = model.predict_proba(X_train)
        X_train_roc = roc_auc_score(y_train, proba_pred, multi_class="ovr")
    elif  svm_multi_class == 2:
        X_train_roc = roc_auc_score(y_train, X_train_pred)
    else:
        X_train_roc = np.nan
    print(f'{name} achieved on validation-set: {round(X_train_acc * 100, 1)}% accuracy, a kappa score of {round(X_train_kappa,2)} & roc score of {round(X_train_roc,2)}.')
    if df_scores.loc[df_scores['Method_Category'] == f"{name} train"].empty:
        print(f"Adding {name} training-set.")
        new_row = {'Method_Category': f"{name} train", 'Accuracy': X_train_acc, 'Kappa': X_train_kappa, 'Roc': X_train_roc}
        df_scores = df_scores.append(new_row, ignore_index = True)
        df_scores.to_csv("scores/nondeep.csv", index=False)
    else:
        print(f"Updating {name} test-set.")
        df_scores.loc[df_scores['Method_Category'] == f"{name} train"] = f"{name} train", X_train_acc, X_train_kappa, X_train_roc
        df_scores.to_csv("scores/nondeep.csv", index=False)
    return(df_scores)

def add_pred_score(name, pred, df_scores): # ROC requires multiclass parameters.
    X_test_pred = pred

    # Obtain and check accuracy on test data
    X_test_acc = accuracy_score(X_test_pred, y_test)
    # X_test_acc = balanced_accuracy_score(X_test_pred, y_test, sample_weight = X_train_weights_by_class)
    X_test_kappa = cohen_kappa_score(y_test, X_test_pred)
    X_test_roc = roc_auc_score(y_test, X_test_pred)
    print(f'{name} achieved on test-set: {round(X_test_acc * 100, 1)}% accuracy, a kappa score of {round(X_test_kappa,2)} & roc score of {round(X_test_roc,2)}.')
    if df_scores.loc[df_scores['Method_Category'] == f"{name} test"].empty:
        print(f"Adding {name} test-set.")
        new_row = {'Method_Category': f"{name} test", 'Accuracy': X_test_acc, 'Kappa': X_test_kappa, 'Roc': X_test_roc}
        df_scores = df_scores.append(new_row, ignore_index = True)
        df_scores.to_csv("scores/nondeep.csv", index=False)
    else:
        print(f"Updating {name} test-set.")
        df_scores.loc[df_scores['Method_Category'] == f"{name} test"] = f"{name} test", X_test_acc, X_test_kappa, X_test_roc
        df_scores.to_csv("scores/nondeep.csv", index=False)
    

# ------------------------------- CC -----------------------------------------
#%% Load numpy array from npy file - CC
X_train = load('data/X_train_CC.npy')
y_train = load('data/y_train_CC.npy')
X_test = load('data/X_test_CC.npy')
y_test = load('data/y_test_CC.npy')
X_val = load('data/X_val_CC.npy')
y_val = load('data/y_val_CC.npy')

X_train = np.concatenate((X_train, X_val), axis=0)
y_train = np.concatenate((y_train, y_val), axis=0)

# Scaling data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


#%% SCM CC -------------------------------------------------------------------
#%% Load results
svm_CC = joblib.load("data/q12svmCC.pkl")

df_scores = add_model_score("SVM CC", svm_CC, df_scores, 1)

#%% RF CC --------------------------------------------------------------------
rf_CC = joblib.load('data/q12rfCC_roc_auc.pkl')

df_scores = add_model_score("RF CC", rf_CC, df_scores, 1)

#%% Boosting CC --------------------------------------------------------------
boosting_CC = joblib.load('data/GB_CC_roc_auc.pkl')

df_scores = add_model_score("Boosting CC", boosting_CC, df_scores, 1)

# #%% CNN CC -------------------------------------------------------------------
# CNN_CC = load("predictions/y_hat_CC.npy")

# df_scores = add_pred_score("CNN CC", CNN_CC, df_scores)

# ------------------------------- D ------------------------------------------
#%% Load numpy array from npy file - D
X_train = load('data/X_train_D.npy')
y_train = load('data/y_train_D.npy')
X_test = load('data/X_test_D.npy')
y_test = load('data/y_test_D.npy')
X_val = load('data/X_val_D.npy')
y_val = load('data/y_val_D.npy')

X_train = np.concatenate((X_train, X_val), axis=0)
y_train = np.concatenate((y_train, y_val), axis=0)

# Scaling data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

#%% SCM D --------------------------------------------------------------------
svm_D = joblib.load("data/q12svmD.pkl")

df_scores = add_model_score("SVM D", svm_D, df_scores, 2)

#%% RF D ---------------------------------------------------------------------
rf_D = joblib.load('data/q12rf_D.pkl')

df_scores = add_model_score("RF D", rf_D, df_scores, 3)

#%% Boosting D ---------------------------------------------------------------
boosting_D = joblib.load('data/q12gb_D.pkl')

df_scores = add_model_score("Boosting D", boosting_D, df_scores, 3)

# ------------------------------- Y -----------------------------------------
#%% Load numpy array from npy file - Y
X_train = load('data/X_train_Y.npy')
y_train = load('data/y_train_Y.npy')
X_test = load('data/X_test_Y.npy')
y_test = load('data/y_test_Y.npy')
X_val = load('data/X_val_Y.npy')
y_val = load('data/y_val_Y.npy')

X_train = np.concatenate((X_train, X_val), axis=0)
y_train = np.concatenate((y_train, y_val), axis=0)

# Scaling data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

#%% SCM Y --------------------------------------------------------------------
svm_Y = joblib.load("data/q12svmY.pkl")

df_scores = add_model_score("SVM Y", svm_Y, df_scores, 1)

#%% RF Y ---------------------------------------------------------------------
rf_Y = joblib.load('data/q12rf_Y.pkl')

df_scores = add_model_score("RF Y", rf_Y, df_scores, 3)

#%% Boosting Y ---------------------------------------------------------------
boosting_Y = joblib.load('data/q12gb_Y.pkl')

df_scores = add_model_score("Boosting Y", boosting_Y, df_scores, 3)
