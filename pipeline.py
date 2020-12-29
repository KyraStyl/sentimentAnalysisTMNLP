#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 14:30:30 2020

@author: kyrastyl
"""

import sys
import pandas as pd
import preprocessing as pp
import emoji

args = sys.argv

print(len(args))

if(len(args)<1):
    print("No arguments passed!\nTry again.")
    sys.exit()
else:
    file = args[0]
    data = pd.read_csv(file)
    o=pp.removeSpecialChars("St=0ell-11a2")
    print(o)
    o=pp.dealWithContractions("I didn't like this movie")
    print(o)
    o=pp.dealWithEmoji("I loved this song!! <3")
    print(o)
    o=pp.dealWithEmoji(pp.fixMisspells("I looooooveood this song! <3"))
    print(o)
    o=pp.removeStopWords("I liked and loved me myself and I")
    print(o)
    before="I will send a "+emoji.emojize(":love_letter:")+" to you with all my <3 :) :D :P ."
    print(before)
    o=pp.dealWithEmoji(before)
    print(o)