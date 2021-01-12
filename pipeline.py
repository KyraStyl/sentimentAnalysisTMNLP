#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 14:30:30 2020

@author: kyrastyl
"""

import sys
import pandas as pd
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
    # import file
    file = args[0]
    #file = "84_clean.csv"
    # load it
    dataloader = dl.DataLoader(file)
    X,y = dataloader.load_dataset()
    #preprocess tweets
    X = [' '.join(pp.preprocess(tweet)) for tweet in X]
    #re-set the tweets in dataloader
    dataloader.set_X(X)
    # split to train and test
    X_train, X_test, y_train, y_test = dataloader.train_test_split(0.7)
    
    
    