# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 00:28:59 2021

@author: AKNOUCHE Anis
"""

import pickle
import numpy as np
from Utils import getCVTextsAndLabels, tokenize, vectorization
from sklearn import model_selection 
from Classifieurs import SVM, KNN, Aleatoire, Majoritaire, MultinomialNaiveBayes


data_cv = pickle.load( open( "CV_5000_PLDAC.pkl", "rb" ) )
print(type(data_cv))

"""
#prise en main des données 

print("data_cv est une liste de dictionnaires")
print("les clés du dict CV :",data_cv[1].keys(),"\n")
print("jobs est une liste de dictionnaires")
print("les clés du dict jobs:",data_cv[0].get('jobs')[0].keys(),"\n")
print("les skills est une liste de chaines de caractères représentant les compétences")
print(data_cv[2].get('skills'),"\n")
print("edu est une liste de dictionnaires")
print("les clés du dict edu:",data_cv[2].get('edu')[0].keys(),"\n")
print("industry est une chaine de caractères")
print(data_cv[2].get('industry'))
print(data_cv[1].get('jobs')[0].values())
"""

textsCV, labelsCV=getCVTextsAndLabels(data_cv)
textsCV=tokenize(textsCV)
dataSet_x, dataLabel_y, labels=vectorization(textsCV, labelsCV)


tmp=np.sum(dataSet_x,axis=0)
#Liste contenant des listes d'indices des mots avec 0 ou 1 occurence seulement dans le dataset réduit
listeIndOcc=[]
for i in range(4):
    listeIndOcc.append(list(np.where(tmp==i)[0]))
listeIndOcc=listeIndOcc[0]+listeIndOcc[1]+listeIndOcc[2]+listeIndOcc[3]
dataSet_x=np.delete(dataSet_x,listeIndOcc,axis=1)
#découpage des données d'entrainements et tests
x_train, x_test, y_train, y_test = model_selection.train_test_split(dataSet_x,dataLabel_y, train_size=0.80, test_size=0.20, random_state=101)


"""
#Test d'un classifieur SVM

svm=SVM(C=2,kernel='poly',degree=2)
svm.fit(x_train, y_train)

print('Accuracy (poly Kernel): ', "%.2f" % (svm.accuracy(svm.predict(x_test),y_test)*100))
print('F1 (poly Kernel): ', "%.2f" % (svm.fMeasure(svm.predict(x_test),y_test)*100))
"""

"""
#Test d'un classifieur KNN

knn=KNN(N=5)
knn.fit(x_train, y_train)

print('Accuracy (KNN): ', "%.2f" % (knn.accuracy(knn.predict(x_test),y_test)*100))
print('F1 (poly (KNN): ', "%.2f" % (knn.fMeasure(knn.predict(x_test),y_test)*100))
"""

"""
#test d'un classifieur aleatoire

aleatoire=Aleatoire(labels)

#pas de fit

print('Accuracy (KNN): ', "%.2f" % (aleatoire.accuracy(aleatoire.predict(x_test),y_test)*100))
print('F1 (poly (KNN): ', "%.2f" % (aleatoire.fMeasure(aleatoire.predict(x_test),y_test)*100))
 #Obtention d'une accuracy de 5.80% et F1-measure de 6.64%
"""

"""
#test d'un classifieur Majoritaire

majoritaire=Majoritaire(labels)
majoritaire.fit(y_train)


print('Accuracy (Majoritaire): ', "%.2f" % (majoritaire.accuracy(majoritaire.predict(x_test),y_test)*100))
print('F1 (poly (Majoritaire): ', "%.2f" % (majoritaire.fMeasure(majoritaire.predict(x_test),y_test)*100))

 #Obtention d'une accuracy de 14.69% et F1-Measure de 3.76%
"""

#test d'un classifieur mutlinomial naive bayes 

clf=MultinomialNaiveBayes()
clf.fit(x_train, y_train)

print('Accuracy (MultinomialNaiveBayes): ', "%.2f" % (clf.accuracy(clf.predict(x_test),y_test)*100))
print('F1 (MultinomialNaiveBayes): ', "%.2f" % (clf.fMeasure(clf.predict(x_test),y_test)*100))

#Obtention d'une accuracy de 47.53% et F1-Measure de 42.30%

