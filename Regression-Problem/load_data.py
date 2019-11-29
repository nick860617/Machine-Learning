# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 13:52:10 2019

@author: pc
"""

import pandas as pd
import numpy as np

def Read_csv(path_list):
    df_list = []
    np_X_train, np_X_test, np_y_train, np_y_test= [], [], [], []
    
    for i in range(0, len(path_list)):
        df_list.append(np.array(pd.read_csv(path_list[i])))
    
    (X_test, y_test, X_train, y_train) = (df_list[0], 
        df_list[1].reshape(len(df_list[1])), 
        df_list[2], df_list[3].reshape(len(df_list[3])))
    
    for i in range(0, len(y_test)):
        if y_test[i]==1 or y_test[i]==2:
            np_X_test.append(X_test[i])
            np_y_test.append(y_test[i]-1)
    
    for i in range(0, len(y_train)):
        if y_train[i]==1 or y_train[i]==2:
            np_X_train.append(X_train[i])
            np_y_train.append(y_train[i]-1)
    
    (np_X_test, np_y_test, np_X_train, np_y_train) = (np.array(np_X_test),
     np.array(np_y_test), np.array(np_X_train), np.array(np_y_train))
    
    for i in range(0, len(np_X_train[0, :])):
        max_ = max(np_X_train[:, i])
        min_ =  min(np_X_train[:, i])
        if max_==min_:
            max_, min_ = 1, 0
        for j in range(0, len(np_X_train[:, 0])):
            np_X_train[j ,i] = (np_X_train[j ,i]-min_)/(max_ - min_)
        
        for j in range(0, len(np_X_test[:, 0])):
            np_X_test[j ,i] = (np_X_test[j ,i] - min_)/(max_-min_)
    
    return np_X_test, np_y_test, np_X_train, np_y_train

def Read_xls(path, normalize=True):
    X, y = (np.array(pd.read_excel(path))[:, 2:7], 
            np.array(pd.read_excel(path))[:, 7])
    if normalize:
        for i in range(0, len(X[0, :])):
            max_ = max(X[:, i])
            min_ =  min(X[:, i])
            if max_==min_:
                max_, min_ = 1, 0
            for j in range(0, len(X[:, 0])):
                X[j ,i] = (X[j ,i]-min_)/(max_ - min_)
        
    return X, y
