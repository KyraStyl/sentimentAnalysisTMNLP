#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 16:03:36 2020

@author: kyrastyl
"""

import csv
import fasttext
import re
import pandas as pd

class FastTextModel():
    def __init__(self,lr=0.01,epoch=20,wordNgrams=2,dim=20):
        self.lr = lr
        self.epoch = epoch
        self.wordNgrams = wordNgrams
        self.dim = dim
        print("Created Instance Fasttext")

    def mapToLabel(self,score):
        label="NULL"
        if score>=-1 and score<-0.3:
            label="NEGATIVE"
        elif (score<=0.3):
            label="NEUTRAL"
        elif score<=1:
            label="POSITIVE"
        return label
            
    
    def transform_instance(self,score):
        label = "__label__" + self.mapToLabel(score)  
        return label
    
    
    def to_text(self,row):
        text = [row['label'],row['text']]
        return text


    def preprocess(self, X, y, output_file):
        labels = []
        self.outputFile=output_file
        dataset = pd.DataFrame({"text":X,"score":y})
        for index, row in dataset.iterrows():
            labels.append(self.transform_instance(row['score']))
        dataset['label']= labels
        with open(output_file, 'w') as csvoutfile:
            csv_writer = csv.writer(csvoutfile, delimiter=',', lineterminator='\n')
            for index, row in dataset.iterrows():
                if row['label'] in ['__label__POSITIVE','__label__NEGATIVE','__label__NEUTRAL'] and row['text']!='':
                    csv_writer.writerow(self.to_text(row))
              
                
    def fit(self,X_train,y_train):
        self.preprocess(X_train,y_train,"training_output.csv")
        hyper_params = {"lr":self.lr,"epoch":self.epoch,"wordNgrams":self.wordNgrams,"dim":self.dim}
        model = fasttext.train_supervised(input=self.outputFile, **hyper_params)
        self.model = model
            
    
    def predict(self,X_test):
        predictions = []
        for row in X_test:
            pred = re.sub("\,[A-Za-z]+","",self.model.predict(row, k=1)[0][0])
            predictions.append(pred)
        return predictions
    
    
    def score(self,X_test,y_test):
        acc = 0
        predictions = self.predict(X_test)
        y_test = [self.transform_instance(score) for score in y_test]
        for i,p in enumerate(predictions):
            if p == y_test[i]:
                acc +=1
        acc = acc/len(y_test)
        return acc
    
    
    