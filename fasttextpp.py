#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 16:03:36 2020

@author: kyrastyl
"""

import csv
import fasttext
import re

class FastTextPreprocessor():
    def __init__(self,data):
        self.dataset = data


    def mapToLabel(self,score):
        label="NULL"
        if score>=-1 and score<-0.3:
            label="NEGATIVE"
        elif (score<=0.3):
            label="NEUTRAL"
        elif score<=1:
            label="POSITIVE"
        return label
            
    
    def transform_instance(self,row):
        label = "__label__" + self.mapToLabel(row['score'])  
        return label
    
    
    def to_text(self,row):
        text = [row['label'],row['text']]
        return text


    def preprocess(self, output_file):
        self.outputFile = output_file
        labels = []
        for index, row in self.dataset.iterrows():
            labels.append(self.transform_instance(row))
        self.dataset['label']= labels
        with open(output_file, 'w') as csvoutfile:
            csv_writer = csv.writer(csvoutfile, delimiter=',', lineterminator='\n')
            for index, row in self.dataset.iterrows():
                if row['label'] in ['__label__POSITIVE','__label__NEGATIVE','__label__NEUTRAL'] and row['text']!='':
                    csv_writer.writerow(self.to_text(row))

    def train(self,lr=0.01,epoch=20,wordNgrams=2,dim=20):
        hyper_params = {"lr":lr,"epoch":epoch,"wordNgrams":wordNgrams,"dim":dim}
        model = fasttext.train_supervised(input=self.outputFile, **hyper_params)
        self.model = model
        
    """
    def predict(self,testFile):
        predictions = []
        with open(testFile,"r") as f:
            while ((line = f.readline)!= None):
                predictions.append(self.model.predict(line))
        return predictions
    """
    
    def predict(self,testX):
        predictions = []
        for row in testX:
            pred = re.sub("\,[A-Za-z]+","",self.model.predict(row, k=1)[0][0])
            predictions.append(pred)
        return predictions
    
    def score(self,X_test,y_test):
        acc = 0
        predictions = self.predict(X_test)
        for i,p in enumerate(predictions):
            if p == y_test[i]:
                acc +=1
        acc = acc/len(y_test)
        return acc
    
    
    """
    TO TEST FASTTEXT MODEL CTRL+C & CTRL+V THE CODE BELOW, IN PIPELINE.PY
    
    
    X = ["That’s how corona and bad luck will get confused looking for u Morning #ssot https://t.co/1P9vMWmHqk",
         "CoronaVirus and the Challenges for Refugees| GMA Talk #WithRefugees #Refugees #COVID19 #coronavirus https://t.co/VhiXrk8Gw4",
         "#SOS #Delhi Need #Blood Type :  B-positive At : ILBS Hospital, Vasant Kunj Blood Component : Need Plasma B+ve from #COVID…"]
    X = [' '.join(pp.preprocess(tweet)) for tweet in X]
    #print(X)
    
    df = pd.DataFrame({'text':X,'score':[-0.1,0.6,-0.9]})
    ft = ftp.FastTextPreprocessor(df)
    ft.preprocess("testFile.csv")
    ft.train()
    X_test = ["bad luck this year with corona","I feel so happy :) , I am in house all day"]
    y_test = ["__label__NEUTRAL","__label__POSITIVE"]
    print("predictions : ")
    print(ft.predict(X_test))
    acc = ft.score(X_test,y_test)
    print("Accuracy: %.2f " %acc)
    """
    
    