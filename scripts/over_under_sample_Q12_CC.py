# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 14:04:33 2022

@author: A
"""

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