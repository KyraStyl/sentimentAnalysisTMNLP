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


def numP1(num):
    return num+1
args = sys.argv

print(len(args))
print(args)

if(len(args)<1):
    print("No arguments passed!\nTry again.")
    sys.exit()
else:
    # import file
    file = args[0]
    file = "84_clean.csv"
    # load it
    dataloader = dl.DataLoader(file)
    #X,y = dataloader.load_dataset()
    #preprocess tweets
    X = ["That’s how corona and bad luck will get confused looking for u Morning #ssot https://t.co/1P9vMWmHqk",
         "CoronaVirus and the Challenges for Refugees| GMA Talk #WithRefugees #Refugees #COVID19 #coronavirus https://t.co/VhiXrk8Gw4",
         "#SOS #Delhi Need #Blood Type :  B-positive At : ILBS Hospital, Vasant Kunj Blood Component : Need Plasma B+ve from #COVID…"]
    X = [' '.join(pp.preprocess(tweet)) for tweet in X]
    print(X)
    #re-set the tweets in dataloader
    #dataloader.set_X(X)
    # split to train and test
    #X_train, X_test, y_train, y_test = dataloader.train_test_split(0.7)
    
    
    