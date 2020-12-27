#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 14:30:30 2020

@author: kyrastyl
"""

import sys
import pandas as pd
import preprocessing as pp

args = sys.argv

print(len(args))

if(len(args)<1):
    print("No arguments passed!\nTry again.")
    sys.exit()
else:
    file = args[0]
    data = pd.read_csv(file)
    l=pp.removeURLs("abc def")
    print(l)