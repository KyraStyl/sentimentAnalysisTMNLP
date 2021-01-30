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
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix
import re
import plotslib as pl


def score(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    results = {'accuracy': balanced_accuracy_score(y_test, y_pred), 'precision': precision_score(y_test, y_pred, average='macro'), 
    'recall': recall_score(y_test, y_pred, average='macro'), 'f1': f1_score(y_test, y_pred, average='macro')}
    return results



args = sys.argv

print(len(args))
print(args)

if(len(args)<1):
    print("No arguments passed!\nTry again.")
    sys.exit()
else:
    file = args[0]
    
    print("Instantiating model...")

    model = None
    parameters = {}
    load_model = re.sub('[ ]*','',args[1]).lower()
    representation = re.sub('[ ]*','',args[2]).lower()
    
    if load_model=="fasttext":
        parameters = {"lr":0.1,"epoch":500,"wordNgrams":3,"dim":20} #fasttext parameters
        model = ftp.FastTextModel(**parameters)
    elif load_model=="logisticregression":
        parameters = {"penalty":'l2',"C":0.1,"solver":'lbfgs'} #lr parameters
        model = LogisticRegression(**parameters)
    elif load_model=="svm":
        parameters = {"kernel":'rbf', "gamma":'scale', "C":100} #svm parameters
        model = svm.SVC(**parameters)
    elif load_model == "naivebayes":
        parameters = {"alpha":0.1}
        model = MultinomialNB(**parameters)
    else:
        print("WRONG MODEL")
    
    
    # load dataset
    dataloader = dl.DataLoader(file, args[3])
    X_old,y = dataloader.load_dataset(return_X_y=True)
    print("Dataset loaded")

    startPrep = time.time()
    #preprocess tweets
    words = []
    X = []
    for tweet in X_old:
        wordsTemp = pp.preprocess(tweet)
        for word in wordsTemp:
            words.append(word)
            X.append(' '.join(wordsTemp))
            
    stopPrep = time.time()
    
    # create a wordCloud with tweets
    # you can define the max words (default = 2000)
    pl.createWordCloud(words)
    
    #if not fasttext convert score to labels
    if load_model!='fasttext':
        y = [ftp.FastTextModel(quiet=True).mapToLabel(score) for score in y]
    
    #re-set the tweets and scores in dataloader (after changes)
    dataloader.set_X(X,y)
    
    # split to train and test
    print("Train-test split complete")
    X_train, X_test, y_train, y_test = dataloader.train_test_split(X, y, 0.7)
    
    
    # if not fasttext get tfidf or word2vec
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

    if load_model=='fasttext':
        y_train = [model.transform_instance(score) for score in y_train]
        y_test = [model.transform_instance(score) for score in y_test]
    
    print("Score model in training dataset...")
    sc = score(model, X_train, y_train)
    
    pl.plotCOUNTS(y_train, y_test)
    
    print("Metrics of ",load_model," are:", sc)

    print("Score model in test dataset...")
    sc = score(model, X_test, y_test)
    
    print("Metrics of ",load_model," are:", sc)
    
    stopTime = time.time()
    print('Preprocess Took : ', stopPrep - startPrep, 'seconds.')
    print('Train and test Took : ',stopTime - startTime,' seconds.')
