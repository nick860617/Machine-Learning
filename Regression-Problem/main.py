# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 13:46:35 2019

@author: pc
"""
import load_data
import model
import simulation
import numpy as np

from sklearn.metrics import accuracy_score


path_list = ['./csvTestImages 3360x1024.csv', './csvTestLabel 3360x1.csv',
        './csvTrainImages 13440x1024.csv', './csvTrainLabel 13440x1.csv']

xls_path = ['./Real_estate_valuation_dataset.xlsx']

if __name__ == '__main__':
    
    # Problem 1(a)
    X, y = load_data.Read_xls(xls_path[0], normalize=True)
    regressor1 = model.LinearRegression()
    regressor1.fit(X, y)
    (std_e, ts_b, p_values) = simulation.table(regressor1, X, y)

    # Problem 1(b)(c)
    F1, F2 = X[:, 1], X[:, 3]
    XFS = np.vstack((F1, F2)).T
    regressor2 = model.LinearRegression()
    regressor2.fit(XFS, y)
    (std_eFS, ts_bFS, p_valuesFS) = simulation.table(regressor2, XFS, y)
    simulation.plot_(XFS, regressor2)
    
    # Problem 2(a)
    test_X, test_y, train_X, train_y = load_data.Read_csv(path_list)
    clf = model.LogisticRegression()
    clf.fit(train_X, train_y, iter_time=400, learning_rate=0.01)    
    simulation.confusion_matrix(clf, test_y, test_X)
    
    
    