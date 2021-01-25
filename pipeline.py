#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 14:30:30 2020

@author: kyrastyl
"""
import time
import sys
from word2vec import W2V
import preprocessing as pp
import dataloader as dl
import fasttextpp as ftp
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix



args = sys.argv

print(len(args))
print(args)

if(len(args)<1):
    print("No arguments passed!\nTry again.")
    sys.exit()
else:
    #file = args[0]
    file = "data/cleared_05"
    
    print("Instantiating model...")
    
    load_model = "fasttext"
    representation = "tfidf"
    parameters = {}
    if load_model=="fasttext":
        parameters = {"lr":0.2,"epoch":200,"wordNgrams":3,"dim":20} #fasttext parameters
    elif load_model=="logisticregression":
        parameters = {"penalty":'l2',"C":0.1,"solver":'lbfgs'} #lr parameters
    elif load_model=="svm":
        parameters = {"kernel":'rbf', "gamma":'scale', "C":100} #svm parameters
    #parameters = {"alpha":0.1}
 
    model = None
    if load_model == "logisticregression":
        model = LogisticRegression(**parameters)
    elif load_model == "svm":
        model = svm.SVC(**parameters)
    elif load_model == "naivebayes":
        model = MultinomialNB(**parameters)
    elif load_model == "fasttext":
        model = ftp.FastTextModel(**parameters)
    else:
        print("Wrong model")
    
    
    # load dataset
    dataloader = dl.DataLoader(file, 1000)
    X,y = dataloader.load_dataset(return_X_y=True)
    print("Dataset loaded")

    #preprocess tweets
    X = [' '.join(pp.preprocess(tweet)) for tweet in X]
    
    #if not fasttext convert score to labels
    if load_model!='fasttext':
        y = [ftp.FastTextModel(quiet=True).mapToLabel(score) for score in y]
    
    #re-set the tweets and scores in dataloader (after changes)
    dataloader.set_X(X,y)
    
    # split to train and test
    print("Train-test split complete")
    X_train, X_test, y_train, y_test = dataloader.train_test_split(X, y, 0.7)
    
    # if not fasttext get tfidf
    if load_model!='fasttext':
        if representation=='tfidf':
            tfidf, X_train = pp.tfidf_of_corpus(X_train)
            X_test = [pp.tfidfVec(x, tfidf).tolist()[0] for x in X_test]
        elif representation=='word2vec':
            w2v = W2V()
            X_train = [w2v.make_embedding(x) for x in X_train]
            X_test = [w2v.make_embedding(x) for x in X_test]


    startTime = time.time()
    print("Fitting the model...")
    model.fit(X_train,y_train)
    
    def score(model, X_test, y_test):
        y_pred = model.predict(X_test)

        if load_model=='fasttext':
            y_test = [model.transform_instance(score) for score in y_test]
            y_pred = [model.clear_output(score) for score in y_pred]

        print(confusion_matrix(y_test, y_pred))
        results = {'accuracy': balanced_accuracy_score(y_test, y_pred), 'precision': precision_score(y_test, y_pred, average='macro'), 
        'recall': recall_score(y_test, y_pred, average='macro'), 'f1': f1_score(y_test, y_pred, average='macro')}
        return results

    print("Score...")
    sc = score(model, X_train, y_train)
    
    
    print("Metrics of ",load_model," are:", sc)

    print("Score...")
    sc = score(model, X_test, y_test)
    
    
    print("Metrics of ",load_model," are:", sc)
    
    stopTime = time.time()
    print("Execution Took : ",stopTime - startTime," seconds.")
