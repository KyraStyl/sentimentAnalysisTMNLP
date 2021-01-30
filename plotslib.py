#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 00:07:21 2021

@author: kyrastyl
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud
import fasttextpp as ftp

"""
    Creates a WordCloud based on a specific image.
"""
def createWordCloud(words,maxWords=2000):
    
    mask = np.array(Image.open('myCovidBanner.jpg'))
    
    wc = WordCloud(background_color="white", max_words=maxWords, mask=mask)
    clean_string = ','.join(words)
    wc.generate(clean_string)

    plt.figure(figsize=(50,50))
    plt.imshow(wc, interpolation='bilinear')
    plt.title('Covid-19 WordCloud', size=40)
    plt.axis("off")
    plt.show()
    

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
    return [countsNeg, countsNeu, countsPos]


"""
    Creates a paired plot with counts of neutral, negative and positive tweets.
"""
def plotCOUNTS(y_train, y_test):
    labels = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
    means1 = [c for c in calculateCOUNTS(y_train)]
    means2 = [c for c in calculateCOUNTS(y_test)]
    
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, means1, width, label='Train')
    rects2 = ax.bar(x + width/2, means2, width, label='Test')
    
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