# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 01:55:48 2021

@author: DELL VOSTRO
"""

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,f1_score
from sklearn.naive_bayes import MultinomialNB
import numpy as np



class Classifieur():
    
    def fit(dataSet,dataLabel):
        pass
    
    def predict(x):
        pass
    
    def accuracy(self,label_pred, label_test):
        return accuracy_score(label_test,label_pred)
    
    def fMeasure(self,label_pred,y_test):
        return f1_score(y_test, label_pred, average='weighted')
    
    
   
class Aleatoire(Classifieur):
    """ Classifieur Aleatoire """
    
    def __init__(self, labels_unique):
        self.labels_unique=labels_unique
    
        
    def predict(self, data_x):
        
        return np.random.random_integers(0,len(self.labels_unique), data_x.shape[0])
    
    
class Majoritaire(Classifieur):
    """ Classifieur Majoritaire """
    
    def __init__(self,labels_unique):
        """
        -------
        labels_unique : liste de string des domaines industriels
        -------
        """
        
        self.labels_unique=[i for i in range(len(labels_unique))]
        self.classe_majoritaire=""
    
    def fit(self,label_train):
        #Trouver la classe majoritaire
        
        classes={c: label_train.count(c)  for c in self.labels_unique}
        x = sorted(classes, key=(lambda key:classes[key]), reverse=True)
        self.classe_majoritaire=x[0]
        print(classes)
        
        
    def predict(self,data_x):
        n=data_x.shape[0]
        return [ self.classe_majoritaire for i in range(n)]
    
class SVM(Classifieur):
    """ Classifieur SVM """
    
    def __init__(self,C,kernel,degree):
        self.C=C
        self.kernel=kernel
        self.degree=degree
        self.model=svm.SVC(C=self.C,kernel=self.kernel,degree=self.degree)
    
    def fit(self,x_train,y_train):
       
        self.model.fit(x_train,y_train)
        
    def predict(self,dataX):
        return self.model.predict(dataX)
    

class KNN(Classifieur):
    """ Classifieur KNN """
    
    def __init__(self,N=5):
        self.knn = KNeighborsClassifier(n_neighbors=N)
        
    def fit(self,train_data,train_label):
        self.knn.fit(train_data, train_label)
        
    def predict(self, data_x):
        return self.knn.predict(data_x)
    
    
class MultinomialNaiveBayes(Classifieur):
     """ Multinomial Naive Bayes """
     
     def __init__(self):
         self.clf=MultinomialNB()
         
     def fit(self,x,y):
         self.clf.fit(x,y)
        
     def predict(self,x):
         return self.clf.predict(x)
     
    
    
    
    
    

    