# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 14:02:25 2019

@author: pc
"""
import numpy as np

class LinearRegression():
    def __init__(self):
        self.w = None
        
    def fit(self, X, y, intercept=1):
        if intercept:
            intercept = np.ones((X.shape[0], 1))
            X = np.hstack((intercept, X))
        self.w = np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), y)

class LogisticRegression():
    def __init__(self, intercept=True):
        self.w = None
        self.intercept = intercept
        self.loss = []
        
    def fit(self, X, y, iter_time=3000, learning_rate=0.01):
        if self.intercept:
            intercept = np.ones((X.shape[0], 1))
            X = np.hstack((intercept, X))
            
        if self.w == None:
             self.w = np.zeros(X.shape[1])
        
        for i in range(0, iter_time):
            
            randomize = np.arange(len(X))
            np.random.shuffle(randomize)
            X = X[randomize]
            y = y[randomize]
            
            product = np.dot(X, self.w)
            prediction = self.sigmoid(product)
            error = y-prediction
            gradient = np.dot(X.T, error)
            self.w += learning_rate*gradient
            if i%50==0:
                self.loss.append(self.log_likelihood(X, y))
            
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def log_likelihood(self, X, y):
        pred = np.dot(X, self.w)
        return np.sum(y*pred - np.log(1+np.exp(pred)))
    
    def predict_prob(self, X):
        if self.intercept:
            intercept = np.ones((X.shape[0], 1))
            X = np.hstack((intercept, X))
        return self.sigmoid(np.dot(X, self.w))
    
    def predict(self, X):
        return (self.predict_prob(X)>=0.5)*1.0
    
            