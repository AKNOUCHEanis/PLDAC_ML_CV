# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 01:20:49 2021

@author: DELL VOSTRO
"""
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer


def getCVTextsAndLabels(data_cv):
    """
    Parameters
    ----------
    data_cv : BDD contenant tous les CVs, sous forme d'une liste de dictionnaires'

    Returns
    -------
    textsCV : une liste de tous les textes des CVs
    labelsCV : une liste de tous les labels des CVs (Domaine industriel)

    """
    labelsCV=[]
    textsCV=[]
    for nbCV in range(5000): 
        textCVi=[]
        domaine_industriel=data_cv[nbCV].get('industry')
        if domaine_industriel != '':
            for i in data_cv[nbCV].get('jobs'):
                textCVi=textCVi+list(i.values())
            textCVi=textCVi+data_cv[nbCV].get('skills')
            for i in data_cv[nbCV].get('edu'):
                textCVi=textCVi+list(i.values())
            labelsCV.append(domaine_industriel)
            textCVi=list(map(str, textCVi))
            textsCV.append(' '.join(textCVi))
    return textsCV, labelsCV



def tokenize(textsCV):
    """
    Parameters
    ----------
    textsCV : la liste des textes de tous les CVs

    Returns
    -------
    textCVFinal : la liste de textes de tous les CVs aprés la tokenization

    """

    stop_words = set(stopwords.words('french'))
    
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    textCVFinal=[]
    for cv in textsCV:
        words=tokenizer.tokenize(cv)
        new_sentence=''
        for word in words:
            if word not in stop_words:
                new_sentence=new_sentence+' '+word.upper()
            
        textCVFinal.append(new_sentence)
        
    return textCVFinal

def vectorization(textsCV,labelsCV):
    """
    Parameters
    ----------
    textsCV : liste des textes des CVs
    labelsCV : Liste des labels des CVs

    Returns
    -------
    dataSet_x : une matrice où les lignes représentent les documents et les colonnes les mots
    et contient la fréquence d'apparition de chaque mot dans le document correspondant'
    dataLabel_y : un tableau des labels des CVs (Domaine industriel) numérisés
    label_y : contient une liste de tous les domaines industriels

    """
    vectorizer=CountVectorizer()
    vector=vectorizer.fit_transform(textsCV)
    dataSet_x=vector.toarray()
    dataLabel_y=np.array(labelsCV)
    label_y=list(np.unique(dataLabel_y))
    dataLabel_y_num=[label_y.index(y) for y in dataLabel_y]
    
    return dataSet_x, dataLabel_y_num, label_y
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    