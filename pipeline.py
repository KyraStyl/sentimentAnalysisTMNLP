#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 14:30:30 2020

@author: kyrastyl
"""
import time
import sys
#from word2vec import W2V
import preprocessing as pp
import dataloader as dl
import fasttextpp as ftp
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix
import re
import numpy as np
import matplotlib.pyplot as plt


def calculateCOUNTS(y):
    countsPos=0
    countsNeu=0
    countsNeg=0
    
    for target in y:
        target = ftp.FastTextModel(quiet=True).mapToLabel(target)
        if target=="POSITIVE":
            countsPos+=1
        elif target=="NEUTRAL":
            countsNeu+=1
        elif target=="NEGATIVE":
            countsNeg+=1
        #else:
           # print("Unknown Target!")
    return [countsNeg, countsNeu, countsPos]

def plotCOUNTS(y_train, y_test,factor):
    labels = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
    means1 = [c*factor for c in calculateCOUNTS(y_train)]
    means2 = [c*factor for c in calculateCOUNTS(y_test)]
    
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, means1, width, label='Train')
    rects2 = ax.bar(x + width/2, means2, width, label='Test')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Counts')
    ax.set_title('Counts by set and class')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    
    plt.show()
    
def score(model, X_test, y_test):
    y_pred = model.predict(X_test)

    if load_model=='fasttext':
        y_pred = [model.clear_output(score) for score in y_pred]

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
    #file = args[0]
    file = "dataset_stella.csv"
    
    print("Instantiating model...")
    
    load_model = "Fasttext"
    representation = "tfidf"
    model = None
    parameters = {}
    load_model = re.sub('[ ]*','',load_model).lower()
    
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
    dataloader = dl.DataLoader(file, 500000)
    X,y = dataloader.load_dataset(return_X_y=True)
    print("Dataset loaded")

    #preprocess tweets
    #X = [' '.join(pp.preprocess(tweet)) for tweet in X]
    
    #if not fasttext convert score to labels
    #if load_model!='fasttext':
     #   y = [ftp.FastTextModel(quiet=True).mapToLabel(score) for score in y]
    
    #re-set the tweets and scores in dataloader (after changes)
    dataloader.set_X(X,y)
    
    # split to train and test
    print("Train-test split complete")
    X_train, X_test, y_train, y_test = dataloader.train_test_split(X, y, 0.7)

    plotCOUNTS(y_train, y_test, 2)    
    
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

    if load_model=='fasttext':
        y_train = [model.transform_instance(score) for score in y_train]
        y_test = [model.transform_instance(score) for score in y_test]
    
    print("Score...")
    sc = score(model, X_train, y_train)
    
    print("Metrics of ",load_model," are:", sc)

    print("Score...")
    sc = score(model, X_test, y_test)
    
    print("Metrics of ",load_model," are:", sc)
    
    stopTime = time.time()
    print("Execution Took : ",stopTime - startTime," seconds.")
