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
from sklearn.metrics import accuracy_score, cohen_kappa_score, make_scorer, roc_auc_score
from sklearn import ensemble # ensemble instead of tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import seaborn
#from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint
from imblearn.ensemble import BalancedRandomForestClassifier
from numpy import mean
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from collections import Counter
import imblearn
from imblearn.ensemble import BalancedRandomForestClassifier
kappa_scorer = make_scorer(cohen_kappa_score)
# Set path to parrent location of current file
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
os.chdir("../")

#%% Initate pandas dataframe for comparing results
if os.path.isfile('scores/nondeep.csv'):
    df_scores = pd.read_csv('scores/nondeep.csv')
else:
    df_scores = pd.DataFrame(columns=['Method_Category', 'Accuracy', 'Kappa', 'Roc'])
    os.makedirs('scores/', exist_ok=True)
    df_scores.to_csv("scores/nondeep.csv", index=False)


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

#%% Anders
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
svm_poly_best = svm.SVC(kernel='rbf', C = 100, gamma = 'auto')

# Use both training and validation data to fit it (np.concatenate "stacks" the array like rbind in R)
svm_poly_best.fit(np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val]))

#%% Question 1.2 Problem solving: CC SVM Best model performance
# Predictions
X_test_pred = svm_poly_best.predict(X_test)
X_val_pred = svm_poly_best.predict(X_val)
X_train_pred = svm_poly_best.predict(X_train)

# Obtain and check accuracy on test data
X_test_acc = accuracy_score(X_test_pred, y_test)
X_test_kappa = cohen_kappa_score(y_test, X_test_pred)
X_test_roc = roc_auc_score(y_test, X_test_pred)
print(f'SVM for Y achieved on test-set: {round(X_test_acc * 100, 1)}% accuracy, a kappa score of {round(X_test_kappa,2)} & roc score of {round(X_test_roc,2)}.')
if df_scores.loc[df_scores['Method_Category'] == "SVM CC test"].empty:
    print("Adding SVM Y test-set.")
    new_row = {'Method_Category': "SVM CC test", 'Accuracy': X_test_acc, 'Kappa': X_test_kappa, 'Roc': X_test_roc}
    df_scores = df_scores.append(new_row, ignore_index = True)
    df_scores.to_csv("scores/nondeep.csv", index=False)
else:
    print("Updating SVM Y test-set.")
    df_scores.loc[df_scores['Method_Category'] == "SVM CC test"] = "SVM CC test", X_test_acc, X_test_kappa, X_test_roc
    df_scores.to_csv("scores/nondeep.csv", index=False)

# Obtain and check accuracy on validation data
X_val_acc = accuracy_score(X_val_pred, y_val)
X_val_kappa = cohen_kappa_score(y_val, X_val_pred)
X_val_roc = roc_auc_score(y_val, X_val_pred)
print(f'SVM for Y achieved on validation-set: {round(X_val_acc * 100, 1)}% accuracy, a kappa score of {round(X_val_kappa,2)} & roc score of {round(X_val_roc,2)}.')
if df_scores.loc[df_scores['Method_Category'] == "SVM CC val"].empty:
    print("Adding SVM Y validation-set.")
    new_row = {'Method_Category': "SVM CC val", 'Accuracy': X_val_acc, 'Kappa': X_val_kappa, 'Roc': X_val_roc}
    df_scores = df_scores.append(new_row, ignore_index = True)
    df_scores.to_csv("scores/nondeep.csv", index=False)
else:
    print("Updating SVM Y test-set.")
    df_scores.loc[df_scores['Method_Category'] == "SVM CC val"] = "SVM CC val", X_val_acc, X_val_kappa, X_val_roc
    df_scores.to_csv("scores/nondeep.csv", index=False)

# Obtain and check accuracy on training data
X_train_acc = accuracy_score(X_train_pred, y_train)
X_train_kappa = cohen_kappa_score(y_train, X_train_pred)
X_train_roc = roc_auc_score(y_train, X_train_pred)
print(f'SVM for Y achieved on validation-set: {round(X_train_acc * 100, 1)}% accuracy, a kappa score of {round(X_train_kappa,2)} & roc score of {round(X_train_roc,2)}.')
if df_scores.loc[df_scores['Method_Category'] == "SVM CC train"].empty:
    print("Adding SVM Y training-set.")
    new_row = {'Method_Category': "SVM CC train", 'Accuracy': X_train_acc, 'Kappa': X_train_kappa, 'Roc': X_train_roc}
    df_scores = df_scores.append(new_row, ignore_index = True)
    df_scores.to_csv("scores/nondeep.csv", index=False)
else:
    print("Updating SVM Y test-set.")
    df_scores.loc[df_scores['Method_Category'] == "SVM CC train"] = "SVM CC train", X_train_acc, X_train_kappa, X_train_roc
    df_scores.to_csv("scores/nondeep.csv", index=False)

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
min_samples_leaf = [1, 2, 4, 10, 20, 50]
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

random_grid_RF_bal = {'n_estimators': n_estimators,
                      'max_features': max_features,
                      'max_depth': max_depth,
                      'min_samples_split': min_samples_split,
                      'bootstrap': bootstrap,
                      'min_samples_leaf': min_samples_leaf}

random_grid_GB = {'max_depth': max_depth,
                  'min_samples_leaf': min_samples_leaf,
                  'learning_rate': learning_rate}

pprint(random_grid_RF)
pprint(random_grid_GB)
pprint(random_grid_RF_bal)
#%% Question 1.2 Problem solving: CC RF - Anders
#%% Standard model without tuning
# Initialize
rf_std_CC = ensemble.RandomForestClassifier(random_state=(42))
# Fit
rf_std_CC .fit(X_train, y_train)
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

#%% Save model 
joblib.dump(rf_std_CC, 'data/q12rfCC_std.pkl')
#%% load model 
rf_std_CC = joblib.load('data/q12rfCC_std.pkl')
#%% optimize model and save it. 
metrics = [kappa_scorer, 'roc_auc']
for i in metrics: 
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf_CC = ensemble.RandomForestClassifier(random_state=42)
    # Random search of parameters, using 3 fold cross validation, 
    # search across 10 (could be more) different combinations, use all available cores, and evaluate the performance with kappa and roc_auc
    rf_CC = RandomizedSearchCV(rf_CC, random_grid_RF, n_iter = 10, cv = 3, verbose=2, random_state=42, n_jobs = -1, scoring=i) 
    # Fit the model
    rf_CC.fit(np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val]))
    joblib.dump(rf_CC, f'data/q12rfCC_{i}.pkl')
    
#%% load model ROC_AUC
rf_CC_ROC = joblib.load('data/q12rfCC_roc_auc.pkl')
print(rf_CC_ROC.best_params_)

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

#%% load model Kappa
rf_CC_kappa = joblib.load('data/q12rfCC_make_scorer(cohen_kappa_score).pkl')
print(rf_CC_kappa.best_params_)

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

#%%
# Initialize
for i in metrics: 
    rf_CC_bal = imblearn.ensemble.BalancedRandomForestClassifier(random_state=42)
    rf_CC_bal = RandomizedSearchCV(rf_CC_bal, random_grid_RF_bal, n_iter = 10, cv = 3, verbose=2, random_state=42, n_jobs = -1, scoring=i)
    # Fit
    rf_CC_bal.fit(np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val]))
    joblib.dump(rf_CC_bal, f'data/q12rfCC_bal_{i}.pkl')
#%%
# Predict
rf_CC_bal_kappa = joblib.load('data/q12rfCC_bal_make_scorer(cohen_kappa_score).pkl')
print(rf_CC_bal_kappa.best_params_)
y_test_hat_CC_bal_kappa = rf_CC_bal_kappa.predict(X_test)
# accuracy and kappa score for evaluating performance
accuracy = accuracy_score(y_test, y_test_hat_CC_bal_kappa)
kappa = cohen_kappa_score(y_test, y_test_hat_CC_bal_kappa)
roc_auc = roc_auc_score(y_test, y_test_hat_CC_bal_kappa)
print(f'''RF with tuned settings achieved {round(accuracy * 100, 1)}% accuracy a kappa score of {round(kappa,3)} and roc_auc of {round(roc_auc,3)}.''')

# confusion matrix
df_confusion = pd.crosstab(y_test, y_test_hat_CC_bal_kappa, rownames=['Actual'], colnames=['Predicted'],dropna=False)
plot_confusion_matrix(df_confusion)

#%%
# Predict
rf_CC_bal_ROC = joblib.load('data/q12rfCC_bal_roc_auc.pkl')
print(rf_CC_bal_ROC.best_params_)
y_test_hat_CC_bal_ROC = rf_CC_bal_ROC.predict(X_test)
# accuracy and kappa score for evaluating performance
accuracy = accuracy_score(y_test, y_test_hat_CC_bal_ROC)
kappa = cohen_kappa_score(y_test, y_test_hat_CC_bal_ROC)
roc_auc = roc_auc_score(y_test, y_test_hat_CC_bal_ROC)
print(f'''RF with tuned settings achieved {round(accuracy * 100, 1)}% accuracy a kappa score of {round(kappa,3)} and roc_auc of {round(roc_auc,3)}.''')
# confusion matrix
df_confusion = pd.crosstab(y_test, y_test_hat_CC_bal_ROC, rownames=['Actual'], colnames=['Predicted'],dropna=False)
plot_confusion_matrix(df_confusion)

#%% oversampling and under sampling 
# on imbalanced dataset with SMOTE oversampling and random undersampling


#%% oversampling and under sampling 
# decision tree  on imbalanced dataset with SMOTE oversampling and random undersampling

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
#%%
for i in metrics:
    rf_CC_over_under = ensemble.RandomForestClassifier(random_state=42)
    # Random search of parameters, using 3 fold cross validation, 
    # search across 10 different combinations, use all available cores, and evaluate the performance with kappa
    rf_CC_over_under = RandomizedSearchCV(rf_CC_over_under, random_grid_RF, n_iter = 10, cv = 3, verbose=2, random_state=42, n_jobs = -1, scoring=i)
    # Fit the model
    rf_CC_over_under.fit(X_train1,y_train1)
    joblib.dump(rf_CC_over_under, f'data/q12rfCC_over_under{i}.pkl')
#%%
rf_CC_over_under_kappa = joblib.load('data/q12rfCC_over_undermake_scorer(cohen_kappa_score).pkl')
print(rf_CC_over_under_kappa.best_params_)
# predict
y_test_hat_over_under_kappa = rf_CC_over_under_kappa.predict(X_test)
# accuracy and kappa score for evaluating performance
accuracy = accuracy_score(y_test, y_test_hat_over_under_kappa)
kappa = cohen_kappa_score(y_test, y_test_hat_over_under_kappa)
roc_auc = roc_auc_score(y_test, y_test_hat_over_under_kappa)
print(f'''RF with tuned settings achieved {round(accuracy * 100, 1)}% accuracy a kappa score of {round(kappa,3)} and roc_auc of {round(roc_auc,3)}.''')

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

>>>>>>> 469f8b6b315002cd22344903b8bc594621f244ec
# confusion matrix
df_confusion = pd.crosstab(y_test, y_test_hat_over_under_kappa, rownames=['Actual'], colnames=['Predicted'],dropna=False)
plot_confusion_matrix(df_confusion)
#%%
rf_CC_over_under_ROC = joblib.load('data/q12rfCC_over_underroc_auc.pkl')
print(rf_CC_over_under_ROC.best_params_)
# predict
y_test_hat_over_under_ROC = rf_CC_over_under_ROC.predict(X_test)
# accuracy and kappa score for evaluating performance
accuracy = accuracy_score(y_test, y_test_hat_over_under_ROC)
kappa = cohen_kappa_score(y_test, y_test_hat_over_under_ROC)
roc_auc = roc_auc_score(y_test, y_test_hat_over_under_ROC)
print(f'''RF with tuned settings achieved {round(accuracy * 100, 1)}% accuracy a kappa score of {round(kappa,3)} and roc_auc of {round(roc_auc,3)}.''')
# confusion matrix
df_confusion = pd.crosstab(y_test, y_test_hat_over_under_ROC, rownames=['Actual'], colnames=['Predicted'],dropna=False)
plot_confusion_matrix(df_confusion)
#%% Question 1.2 Problem solving: CC B - Anders
GB_CC_std = ensemble.HistGradientBoostingClassifier(random_state=42)
# Fit
GB_CC_std.fit(X_train, y_train)
# save model 
joblib.dump(GB_CC_std, 'data/q12GB_CC_std.pkl')
#%% load model 
GB_CC_std = joblib.load('data/q12GB_CC_std.pkl')
# Predict
y_test_hat_GB_CC_std = GB_CC_std.predict(X_test)
accuracy = accuracy_score(y_test, y_test_hat_GB_CC_std)
kappa = cohen_kappa_score(y_test, y_test_hat_GB_CC_std)
roc_auc = roc_auc_score(y_test, y_test_hat_GB_CC_std)
print(f'''RF with tuned settings achieved {round(accuracy * 100, 1)}% accuracy a kappa score of {round(kappa,3)} and roc_auc of {round(roc_auc,3)}.''')
df_confusion = pd.crosstab(y_test, y_test_hat_GB_CC_std, rownames=['Actual'], colnames=['Predicted'],dropna=False)
plot_confusion_matrix(df_confusion)
#%% tune model
#parameters to tune
random_grid_GB = {'max_depth': max_depth,
                  'min_samples_leaf': min_samples_leaf,
                  'learning_rate': learning_rate,
                  'l2_regularization': [1.5]}   
GB_CC = ensemble.HistGradientBoostingClassifier(random_state=42)
# Random search of parameters, using 3 fold cross validation, 
# search across 10 different combinations, use all available cores, and evaluate the performance with kappa
GB_CC = RandomizedSearchCV(GB_CC, random_grid_GB, n_iter = 10, cv = 3, verbose=2, random_state=42, n_jobs = -1, scoring=kappa_scorer)
# Fit the model
GB_CC.fit(X_train,y_train)
joblib.dump(GB_CC, 'data/q12GB_CC.pkl')
#%%
GB_CC = joblib.load('data/q12GB_CC.pkl')
print(GB_CC.best_params_)
y_test_hat_GB_CC = GB_CC.predict(X_test)
accuracy = accuracy_score(y_test, y_test_hat_GB_CC)
kappa = cohen_kappa_score(y_test, y_test_hat_GB_CC)
roc_auc = roc_auc_score(y_test, y_test_hat_GB_CC)
print(f'''RF with tuned settings achieved {round(accuracy * 100, 1)}% accuracy a kappa score of {round(kappa,3)} and roc_auc of {round(roc_auc,3)}.''')
df_confusion = pd.crosstab(y_test, y_test_hat_GB_CC, rownames=['Actual'], colnames=['Predicted'],dropna=False)
plot_confusion_matrix(df_confusion)
#%%
for i in metrics:
    GB_CC = ensemble.HistGradientBoostingClassifier(random_state=42)
    # Random search of parameters, using 3 fold cross validation, 
    # search across 10 different combinations, use all available cores, and evaluate the performance with kappa
    GB_CC = RandomizedSearchCV(GB_CC, random_grid_GB, n_iter = 10, cv = 3, verbose=2, random_state=42, n_jobs = -1, scoring=i)
    # Fit the model
    GB_CC.fit(X_train,y_train)
    joblib.dump(rf_CC_over_under, f'data/GB_CC_{i}.pkl')
#%% load model kappa
GB_CC_kappa = joblib.load('data/GB_CC_make_scorer(cohen_kappa_score).pkl')
# Predict
print(GB_CC_kappa.best_params_)
y_test_hat_GB_CC_std_kappa = GB_CC_kappa.predict(X_test)
accuracy = accuracy_score(y_test, y_test_hat_GB_CC_std_kappa)
kappa = cohen_kappa_score(y_test, y_test_hat_GB_CC_std_kappa)
roc_auc = roc_auc_score(y_test, y_test_hat_GB_CC_std_kappa)
print(f'''RF with tuned settings achieved {round(accuracy * 100, 1)}% accuracy a kappa score of {round(kappa,3)} and roc_auc of {round(roc_auc,3)}.''')
df_confusion = pd.crosstab(y_test, y_test_hat_GB_CC_std_kappa, rownames=['Actual'], colnames=['Predicted'],dropna=False)
plot_confusion_matrix(df_confusion)
#%%
GB_CC_ROC = joblib.load('data/GB_CC_roc_auc.pkl')
# Predict
print(GB_CC_ROC.best_params_)
y_test_hat_GB_CC_std_ROC = GB_CC_ROC.predict(X_test)
accuracy = accuracy_score(y_test, y_test_hat_GB_CC_std_ROC)
kappa = cohen_kappa_score(y_test, y_test_hat_GB_CC_std_ROC)
roc_auc = roc_auc_score(y_test, y_test_hat_GB_CC_std_ROC)
print(f'''RF with tuned settings achieved {round(accuracy * 100, 1)}% accuracy a kappa score of {round(kappa,3)} and roc_auc of {round(roc_auc,3)}.''')
df_confusion = pd.crosstab(y_test, y_test_hat_GB_CC_std_ROC, rownames=['Actual'], colnames=['Predicted'],dropna=False)
plot_confusion_matrix(df_confusion)
#%%
gbt = imblearn.ensemble.EasyEnsembleClassifier(random_state=42)
gbt.fit(X_train, y_train)
# Predict
y_test_hat = gbt.predict(X_test)
accuracy = accuracy_score(y_test, y_test_hat)
kappa = cohen_kappa_score(y_test, y_test_hat)
roc_auc = roc_auc_score(y_test, y_test_hat)
print(f'''RF with tuned settings achieved {round(accuracy * 100, 1)}% accuracy a kappa score of {round(kappa,3)} and roc_auc of {round(roc_auc,3)}.''')
df_confusion = pd.crosstab(y_test, y_test_hat, rownames=['Actual'], colnames=['Predicted'],dropna=False)
plot_confusion_matrix(df_confusion)

#%% Question 1.2 Problem solving: D
#%% Question 1.2 Problem solving: D SVM
#%% Question 1.2 Problem solving: D SVM gridsearch - Scoring: balanced_accuracy
parameters = {'kernel':['rbf'], 'C':[10, 100], 'gamma':['auto', 'scale'], 'decision_function_shape':['ovr']}
svc = svm.SVC(probability=True)
svm_D = GridSearchCV(svc,
                     parameters,
                     n_jobs=-1, # number of simultaneous jobs (-1 all cores)
                     scoring='balanced_accuracy')
svm_D.fit(np.concatenate((X_train, X_val), axis=0), np.concatenate((y_train, y_val), axis=0))

results = pd.DataFrame(svm_D.cv_results_)
print(results[results['mean_test_score'] == results['mean_test_score'].min()])

#%% Question 1.2 Problem solving: D SVM gridsearch - Save results
joblib.dump(svm_D, 'data/q12svmD.pkl')


#%% Question 1.2 Problem solving: D RF - Anders
# Initialize
rf_D_std = ensemble.RandomForestClassifier(random_state=(42))

# Fit
rf_D_std.fit(X_train, y_train)
joblib.dump(rf_D_std, 'data/q12rf_D_std.pkl')
#%%
rf_D_std = joblib.load('data/q12rf_D_std.pkl')
# Predict
y_test_hat_D_std = rf_D_std.predict(X_test)
accuracy = accuracy_score(y_test, y_test_hat_D_std)
kappa = cohen_kappa_score(y_test, y_test_hat_D_std)
#roc_auc = roc_auc_score(y_test, y_test_hat_D_std)
print(f'''RF with tuned settings achieved {round(accuracy * 100, 1)}% accuracy and a kappa score of {round(kappa,3)}.''')
df_confusion = pd.crosstab(y_test, y_test_hat_D_std, rownames=['Actual'], colnames=['Predicted'],dropna=False)
df_confusion = df_confusion.reindex(columns=[0,1,2,3,4,10], fill_value=0)
plot_confusion_matrix(df_confusion)
#%%
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = ensemble.RandomForestClassifier(random_state=42)
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(rf, random_grid_RF, n_iter = 10, cv = 3, verbose=2, random_state=42, n_jobs = -1, scoring='accuracy')
# Fit the random search model
rf_random.fit(X_train, y_train)
joblib.dump(rf_random, 'data/q12rf_D.pkl')
#%%
rf_D_tuned = joblib.load('data/q12rf_D.pkl')
print(rf_D_tuned.best_params_)
y_test_hat_D_tuned = rf_D_tuned.predict(X_test)
accuracy = accuracy_score(y_test, y_test_hat_D_tuned)
kappa = cohen_kappa_score(y_test, y_test_hat_D_tuned)
print(f'''RF with tuned settings achieved {round(accuracy * 100, 1)}% accuracy and a kappa score of {round(kappa,1)}.''')
df_confusion = pd.crosstab(y_test, y_test_hat_D_tuned, rownames=['Actual'], colnames=['Predicted'],dropna=False)
df_confusion = df_confusion.reindex(columns=[0,1,2,3,4,10], fill_value=0)
plot_confusion_matrix(df_confusion)

#%% Question 1.2 Problem solving: D B - Anders
gbt_D = ensemble.HistGradientBoostingClassifier(random_state=(42), loss='categorical_crossentropy')
# Fit
gbt_D.fit(X_train, y_train)
joblib.dump(gbt_D, 'data/q12gbt_D.pkl')
#%%
gbt_D_std = joblib.load('data/q12gbt_D.pkl')
# Predict
y_test_hat_D = gbt_D_std.predict(X_test)
accuracy = accuracy_score(y_test, y_test_hat_D)
kappa = cohen_kappa_score(y_test, y_test_hat_D)
print(f'''RF with tuned settings achieved {round(accuracy * 100, 1)}% accuracy and a kappa score of {round(kappa,1)}.''')
df_confusion = pd.crosstab(y_test, y_test_hat_D, rownames=['Actual'], colnames=['Predicted'],dropna=False)
plot_confusion_matrix(df_confusion)
#%%
gb_D = ensemble.HistGradientBoostingClassifier(random_state=42,loss='categorical_crossentropy')
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
gb_D = RandomizedSearchCV(gb_D, random_grid_GB, n_iter = 10, cv = 3, random_state=42, n_jobs = -1)
# Fit the random search model
gb_D.fit(X_train, y_train)
joblib.dump(gb_D, 'data/q12gb_D.pkl')
#%%
gb_D_tuned = joblib.load('data/q12gb_D.pkl')
gb_D_tuned.best_params_
y_test_hat_D_Tuned = gb_D_tuned.predict(X_test)
accuracy = accuracy_score(y_test, y_test_hat_D_Tuned)
kappa = cohen_kappa_score(y_test, y_test_hat_D_Tuned)
print(f'''RF with tuned settings achieved {round(accuracy * 100, 1)}% accuracy and a kappa score of {round(kappa,1)}.''')
df_confusion = pd.crosstab(y_test, y_test_hat_D_Tuned, rownames=['Actual'], colnames=['Predicted'],dropna=False)
plot_confusion_matrix(df_confusion)
#%% Question 1.2 Problem solving: Y
metrics.cohen_kappa_score
#%% Question 1.2 Problem solving: Y SVM
#%% Question 1.2 Problem solving: Y SVM gridsearch
parameters = {'kernel':['rbf'], 'C':[1, 10, 100], 'gamma':['auto', 'scale'], 'decision_function_shape':['ovr']}
svc = svm.SVC(probability=True)
svm_Y = GridSearchCV(svc, 
                   parameters,
                   n_jobs=-1, # number of simultaneous jobs (-1 all cores)
                   scoring='accuracy')
svm_Y.fit(np.concatenate((X_train, X_val), axis=0), np.concatenate((y_train, y_val), axis=0))

results = pd.DataFrame(svm_Y.cv_results_)
print(results[results['mean_test_score'] == results['mean_test_score'].min()])

#%% Question 1.2 Problem solving: D SVM gridsearch - Save results
joblib.dump(svm_Y, 'data/q12svmY.pkl')


#%% Question 1.2 Problem solving: Y RF - Anders 
# Initialize
rf_Y_std = ensemble.RandomForestClassifier(random_state=(42))

# Fit
rf_Y_std.fit(X_train, y_train)
joblib.dump(rf_D_std, 'data/q12rf_Y_std.pkl')
#%%
rf_Y_std = joblib.load('data/q12rf_Y_std.pkl')
# Predict
y_test_hat_T_std = rf_Y_std.predict(X_test)
accuracy = accuracy_score(y_test, y_test_hat_T_std)
kappa = cohen_kappa_score(y_test, y_test_hat_T_std)
#roc_auc = roc_auc_score(y_test, y_test_hat_D_std)
print(f'''RF with tuned settings achieved {round(accuracy * 100, 1)}% accuracy and a kappa score of {round(kappa,3)}.''')
df_confusion = pd.crosstab(y_test, y_test_hat_T_std, rownames=['Actual'], colnames=['Predicted'],dropna=False)
plot_confusion_matrix(df_confusion)
#%%
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf_Y = ensemble.RandomForestClassifier(random_state=42)
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_Y = RandomizedSearchCV(rf_Y, random_grid_RF, n_iter = 10, cv = 3, verbose=2, random_state=42, n_jobs = -1, scoring='accuracy')
# Fit the random search model
rf_Y.fit(X_train, y_train)
joblib.dump(rf_random, 'data/q12rf_Y.pkl')
#%%
rf_Y = joblib.load('data/q12rf_Y.pkl')
print(rf_Y.best_params_)
y_test_hat_Y = rf_Y.predict(X_test)
accuracy = accuracy_score(y_test, y_test_hat_Y)
kappa = cohen_kappa_score(y_test, y_test_hat_Y)
print(f'''RF with tuned settings achieved {round(accuracy * 100, 1)}% accuracy and a kappa score of {round(kappa,1)}.''')
df_confusion = pd.crosstab(y_test, y_test_hat_Y, rownames=['Actual'], colnames=['Predicted'],dropna=False)
df_confusion = df_confusion.reindex(columns=[0,1,2,3,4,10], fill_value=0)
plot_confusion_matrix(df_confusion)
#%% Question 1.2 Problem solving: Y B - Anders
gbt_Y = ensemble.HistGradientBoostingClassifier(random_state=(42), loss='categorical_crossentropy')
# Fit
gbt_Y.fit(X_train, y_train)
joblib.dump(gbt_Y, 'data/q12gbt_Y.pkl')
#%%
gbt_Y_std = joblib.load('data/q12gbt_Y.pkl')
# Predict
y_test_hat_Y = gbt_Y_std.predict(X_test)
accuracy = accuracy_score(y_test, y_test_hat_Y)
kappa = cohen_kappa_score(y_test, y_test_hat_Y)
print(f'''RF with tuned settings achieved {round(accuracy * 100, 1)}% accuracy and a kappa score of {round(kappa,1)}.''')
df_confusion = pd.crosstab(y_test, y_test_hat_Y, rownames=['Actual'], colnames=['Predicted'],dropna=False)
plot_confusion_matrix(df_confusion)
#%%
gb_Y = ensemble.HistGradientBoostingClassifier(random_state=42,loss='categorical_crossentropy')
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
gb_Y = RandomizedSearchCV(gb_Y, random_grid_GB, n_iter = 10, cv = 3, random_state=42, n_jobs = -1)
# Fit the random search model
gb_Y.fit(X_train, y_train)
joblib.dump(gb_Y, 'data/q12gb_Y.pkl')
#%%
gb_Y_tuned = joblib.load('data/q12gb_Y.pkl')
gb_Y_tuned.best_params_
y_test_hat_Y_Tuned = gb_Y_tuned.predict(X_test)
accuracy = accuracy_score(y_test, y_test_hat_Y_Tuned)
kappa = cohen_kappa_score(y_test, y_test_hat_Y_Tuned)
print(f'''RF with tuned settings achieved {round(accuracy * 100, 1)}% accuracy and a kappa score of {round(kappa,1)}.''')
df_confusion = pd.crosstab(y_test, y_test_hat_Y_Tuned, rownames=['Actual'], colnames=['Predicted'],dropna=False)
plot_confusion_matrix(df_confusion)
#%% Question 1.2 Performance
print("Calculate and report the method’s performance on the training, validation, and test data.")

# Accuracy metric is bad for evaluating performance.

#%% Question 1.2 Performance: CC
#print(sorted(clf.cv_results_()))
svm_CC_gridsearch_res = joblib.load("data/q12svmCC.pkl")
predictions = svm_CC_gridsearch_res.predict(X_test)

#print(svm_CC_gridsearch_res.best_estimator_)
#print(svm_CC_gridsearch_res.best_params_)
#print(classification_report(y_test, predictions))
#print(svm_CC_gridsearch_res.cv_results_ )

# accuracy and kappa score for evaluating performance
accuracy = accuracy_score(y_test, predictions)
kappa = cohen_kappa_score(y_test, predictions)
roc = roc_auc_score(y_test, predictions)
print(f'SVM for CC achieved: {round(accuracy * 100, 1)}% accuracy, a kappa score of {round(kappa,2)} & roc score of {round(roc,2)}.')

if df_scores.loc[df_scores['Method_Category'] == "SVM CC"].empty:
    print("Adding.")
    new_row = {'Method_Category': "SVM CC", 'Accuracy': accuracy, 'Kappa': kappa, 'Roc': roc}
    df_scores = df_scores.append(new_row, ignore_index = True)
    df_scores.to_csv("scores/nondeep.csv", index=False)
else:
    print("Updating.")
    df_scores.loc[df_scores['Method_Category'] == "SVM CC"] = "SVM CC", accuracy, kappa, roc
    df_scores.to_csv("scores/nondeep.csv", index=False)

#%% Question 1.2 Performance: D
svm_D_gridsearch_res = joblib.load("data/q12svmD.pkl")
predictions = svm_D_gridsearch_res.predict(X_test)
proba_pred = svm_D_gridsearch_res.predict_proba(X_test)
#print(svm_D_gridsearch_res.best_estimator_)
#print(svm_D_gridsearch_res.best_params_)
#print(classification_report(y_test, predictions))

# accuracy and kappa score for evaluating performance
accuracy = accuracy_score(y_test, predictions)
kappa = cohen_kappa_score(y_test, predictions)
roc = roc_auc_score(y_test, proba_pred, multi_class="ovr")
print(f'SVM for D achieved: {round(accuracy * 100, 1)}% accuracy, a kappa score of {round(kappa,2)} & roc score of {round(roc,2)}.')

if df_scores.loc[df_scores['Method_Category'] == "SVM D"].empty:
    print("Adding.")
    new_row = {'Method_Category': "SVM D", 'Accuracy': accuracy, 'Kappa': kappa, 'Roc': roc}
    df_scores = df_scores.append(new_row, ignore_index = True)
    df_scores.to_csv("scores/nondeep.csv", index=False)
else:
    print("Updating.")
    df_scores.loc[df_scores['Method_Category'] == "SVM D"] = "SVM D", accuracy, kappa, roc
    df_scores.to_csv("scores/nondeep.csv", index=False)


#%% Question 1.2 Performance: Y
svm_Y_gridsearch_res = joblib.load("data/q12svmY.pkl")
predictions = svm_Y_gridsearch_res.predict(X_test)
proba_pred = svm_Y_gridsearch_res.predict_proba(X_test)

#print(svm_Y_gridsearch_res.best_estimator_)
#print(svm_Y_gridsearch_res.best_params_)
#print(classification_report(y_test, predictions))

# accuracy and kappa score for evaluating performance
accuracy = accuracy_score(y_test, predictions)
kappa = cohen_kappa_score(y_test, predictions)
roc = roc_auc_score(y_test, proba_pred, multi_class='ovr')
print(f'SVM for Y achieved: {round(accuracy * 100, 1)}% accuracy, a kappa score of {round(kappa,2)} & roc score of {round(roc,2)}.')

if df_scores.loc[df_scores['Method_Category'] == "SVM Y"].empty:
    print("Adding.")
    new_row = {'Method_Category': "SVM Y", 'Accuracy': accuracy, 'Kappa': kappa, 'Roc': roc}
    df_scores = df_scores.append(new_row, ignore_index = True)
    df_scores.to_csv("scores/nondeep.csv", index=False)
else:
    print("Updating.")
    df_scores.loc[df_scores['Method_Category'] == "SVM Y"] = "SVM Y", accuracy, kappa, roc
    df_scores.to_csv("scores/nondeep.csv", index=False)

#%% Question 1.2 Performance-evaluation
print("Does the performance differ between the different sets? If yes, does this surprise you (explain why or why not)?")
print("Evaluation of performance is done in report.")
