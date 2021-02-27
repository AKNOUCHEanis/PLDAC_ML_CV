# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 00:28:59 2021

@author: AKNOUCHE Anis
"""

import pickle
import numpy as np
from Utils import getCVTextsAndLabels, tokenize, vectorization


data_cv = pickle.load( open( "CV_5000_PLDAC.pkl", "rb" ) )
print(type(data_cv))
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

textsCV, labelsCV=getCVTextsAndLabels(data_cv)
textsCV=tokenize(textsCV)
dataSet_x, dataLabel_y, labels=vectorization(textsCV, labelsCV)

print(np.shape(dataSet_x))





