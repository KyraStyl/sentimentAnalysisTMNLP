#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 14:30:30 2020

@author: kyrastyl
"""
import time
import sys
#import pandas as pd
import preprocessing as pp
import dataloader as dl
import fasttextpp as ftp
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB

args = sys.argv

print(len(args))
print(args)

if(len(args)<1):
    print("No arguments passed!\nTry again.")
    sys.exit()
else:
    #file = args[0]
    file = "95_clean.csv"
    
    print("create model")
    
    notfasttext = True
    load_model = "FASTTEXT"
    parameters = {"lr":0.2,"epoch":100,"wordNgrams":3,"dim":20} #fasttext parameters
    #parameters = {"penalty":'l2',"C":1.0,"solver":'lbfgs'} #lr parameters
    #parameters = {"kernel":'sigmoid', "gamma":5, "C":100} #svm parameters
    #parameters = {"alpha":0.1}
    load_model = load_model.lower()
    if load_model == "logisticregression":
        model = LogisticRegression(**parameters)
    elif load_model == "svm":
        model = svm.SVC(**parameters)
    elif load_model == "naivebayes":
        model = MultinomialNB(**parameters)
    elif load_model == "fasttext":
        notfasttext = False
        model = ftp.FastTextModel(**parameters)
    else:
        print("wrong model")
    
    
    # load dataset
    dataloader = dl.DataLoader(file)
    print("load dataset")
    X,y = dataloader.load_dataset(return_X_y=True)
    
    #preprocess tweets
    X = [' '.join(pp.preprocess(tweet)) for tweet in X]
   
    #if not fasttext convert score to labels
    if notfasttext:
        y = [ftp.FastTextModel().mapToLabel(score) for score in y]
    
    #re-set the tweets and scores in dataloader (after changes)
    dataloader.set_X(X,y)
    
    # split to train and test
    print("split to train and test")
    X_train, X_test, y_train, y_test = dataloader.train_test_split(X,y,0.7)
    
    # if not fasttext get tfidf
    if notfasttext:
        tfidf, X_train = pp.tfidf_of_corpus(X_train)
        X_test = [pp.tfidfVec(x,tfidf).toarray().tolist()[0] for x in X_test]


    startTime = time.time()
    print("fit model")
    model.fit(X_train,y_train)
    
    print("score model")
    sc = model.score(X_test, y_test)
    
    print("Accuracy of ",load_model," is:", sc)
    
    stopTime = time.time()
    print("Execution Took : ",stopTime - startTime," seconds.")
