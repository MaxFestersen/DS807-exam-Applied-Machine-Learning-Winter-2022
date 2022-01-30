# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 16:22:54 2022

@author: A
"""
import pandas as pd
import numpy as np
import joblib
#%%
X_test = load('data/X_test_CC.npy')
y_test = load('data/y_test_CC.npy')
#%% Anders
def combine_model(x):
    
    GB_CC = joblib.load('data/q12GB_CC_std.pkl') # best model for CC
    GB_D = joblib.load('data/q12gbt_D.pkl') # best model for D
    GB_Y = joblib.load('data/q12gbt_Y.pkl') # best model for Y
    
    # Get predictions 
    y_hat_CC = GB_CC.predict(x['CC'])
    y_hat_D = GB_D.predict(x['D'])
    y_hat_Y = GB_Y.predict(x['Y'])
    
    # Combine into dataframe
    DF = pd.DataFrame({'y_hat_CC': y_hat_CC, 
                            'y_hat_D': y_hat_D, 
                            'y_hat_Y': y_hat_Y,
                            'True_CC': x['CC'],
                            'True_D': x['D'],
                            'True_Y': x['Y']
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
df = combine_model(X_test)
  