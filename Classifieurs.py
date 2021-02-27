# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 01:55:48 2021

@author: DELL VOSTRO
"""

from sklearn import svm
from sklearn.metrics import accuracy_score,f1_score


class Classifieur():
    
    def fit(dataSet,dataLabel):
        pass
    
    def predict(x):
        pass
    
    def accuracy(label_pred, label_test):
        return accuracy_score(label_test,label_pred)
    
    def fMeasure(label_pred,y_test):
        return f1_score(y_test, label_pred, average='weighted')
    
class Svm(Classifieur):
    """ Classifieur SVM
    """
    
    def __init__(self,C,kernel,degree):
        self.C=C
        self.kernel=kernel
        self.degree=degree
        self.model=svm.SVC(C=self.C,kernel=self.kernel,degree=self.degree)
    
    def fit(self,x_train,y_train):
       
        self.model.fit(x_train,y_train)
        
    def predict(self,dataX):
        return self.model.predict(dataX)
    

   
    
    
   
    
    

    