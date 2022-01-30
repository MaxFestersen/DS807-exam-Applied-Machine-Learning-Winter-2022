# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 16:22:54 2022

@author: Anders
"""
import pandas as pd
import joblib
from numpy import load
from collections import Counter
#%% Anders
df_new_data = pd.read_csv('labels/Digit_String_2_labels_formatted.csv')
y_new_data = df_new_data[['CC', 'D', 'Y']]
X_unseen = load('data/X_unseen.npy')
#%% Anders
def combine_model(x):
    
    GB_CC = joblib.load('data/q12GB_CC_std.pkl') # best model for CC
    GB_D = joblib.load('data/q12gbt_D.pkl') # best model for D
    GB_Y = joblib.load('data/q12gbt_Y.pkl') # best model for Y
    
    # Get predictions 
    y_hat_CC = GB_CC.predict(x)
    y_hat_D = GB_D.predict(x)
    y_hat_Y = GB_Y.predict(x)
    
    # Combine into dataframe
    DF = pd.DataFrame({'y_hat_CC': y_hat_CC, 
                            'y_hat_D': y_hat_D, 
                            'y_hat_Y': y_hat_Y,
                            'True_CC': y_new_data['CC'],
                            'True_D': y_new_data['D'],
                            'True_Y': y_new_data['Y']
                            }, 
                           columns=['y_hat_CC', 
                                    'y_hat_D', 
                                    'y_hat_Y',
                                    'True_CC',
                                    'True_D',
                                    'True_Y'
                                    ])

    return DF
#%%
df = combine_model(X_unseen)    
#%%
for i in ['CC','D','Y']:
    df[f'Score_{i}'] = df[f'True_{i}']==df[f'y_hat_{i}']
    df[f'Score_{i}'] = df[f'Score_{i}'].astype(int)
df['total_score'] = (df['Score_CC']+df['Score_D']+df['Score_Y'])/3   
#%%
print(df['total_score'].sum())
print(Counter(df['total_score'])) 
print(Counter(df['Score_CC']))
print(Counter(df['Score_D']))
print(Counter(df['Score_Y']))