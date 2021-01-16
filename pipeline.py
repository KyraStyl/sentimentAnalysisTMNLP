#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 14:30:30 2020

@author: kyrastyl
"""

import sys
#import pandas as pd
import preprocessing as pp
import dataloader as dl
import fasttextpp as ftp

args = sys.argv

print(len(args))
print(args)

if(len(args)<1):
    print("No arguments passed!\nTry again.")
    sys.exit()
else:
    #file = args[0]
    file = "84_clean.csv"
    
    # load it
    dataloader = dl.DataLoader(file)
    X,y = dataloader.load_dataset()
    
    #preprocess tweets
    X = [' '.join(pp.preprocess(tweet)) for tweet in X]
    
    #re-set the tweets in dataloader
    dataloader.set_X(X)
    
    # split to train and test
    print("split to train and test")
    X_train, X_test, y_train, y_test = dataloader.train_test_split(0.7)
    
    print("load dataset")
    dataset = dataloader.load_dataset(return_X_y=False)
    
    print("create model")
    ft = ftp.FastTextModel()
    
    print("fit model")
    ft.fit(X_train,y_train)
    
    print("score model")
    sc = ft.score(X_test,y_test)
    print("Accuracy: ",sc)
    