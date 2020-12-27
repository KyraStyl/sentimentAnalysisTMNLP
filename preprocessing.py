#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 14:25:29 2020

@author: kyrastyl

This is a python script which works as a library.

Write here all preprocessing functions.

"""

def removeURLs(inputStr):
    """
    :param inputStr: the input string
    :return outputStr: inputStr without urls
    """
    outputStr = inputStr
    return len(outputStr)
    
def removeSpecialChars(inputStr):
    """
    :param inputStr: the input string
    :return outputStr: inputStr without special characters
    
    Remove special characters from input string.
    Special characters: #,@,!,-, etc.
    """
    outputStr = inputStr
    return len(outputStr)

def removeStopWords(inputStr):
    """
    :param inputStr: the input string
    :return outputStr: inputStr without stopwords
    
    Remove all stopwords using library nltk
    (Stopwords: "a","the","and", etc.)
    """
    outputStr = inputStr
    return len(outputStr)

def dealWithEmoji(inputStr):
    """
    :param inputStr: the input string
    :return outputStr: inputStr without emoji
    
    Map each emoji from inputStr to its description?
    """
    outputStr = inputStr
    return len(outputStr)

def dealWithRTs(inputStr):
    """
    :param inputStr: the input string
    :return outputStr: inputStr without re-tweets
    
    For each re-tweeted tweet, keep only the initial tweet.
    e.g. 
    for -> "id": 1240727816393469952, "id_str": "1240727816393469952",
    "full_text": "RT @CandiceBenbow: This is Generation Z. 
    \n\nI want to name that because folks love calling everybody 40 and 
    under Millennials.\n\nMillennials\u2026",
    
    keep only -> "This is Generation Z ...etc."
    """
    outputStr = inputStr
    return len(outputStr)

def dealWithNegation(inputStr):
    """
    :param inputStr: the input string
    :return outputStr: inputStr with negation tokens
    
    For each sentence, if a negative token is presence (e.g. "no"/"not"/etc.)
    replace each successive <token> with <not_token>
    
    e.g. a simple example
    
    EXAMPLE-> "I get corona, I get corona. At the end of the day, 
    I'm not gonna let it stop me from partying."
    
    BECOMES-> "I get corona, I get corona. At the end of the day, 
    I'm  not_gonna not_let not_it not_stop not_me not_from not_partying."
    """
    outputStr = inputStr
    return len(outputStr)

def tfidfVec(inputStr):
    """
    :param inputStr: the input string
    :return outputStr: inputStr to TF_IDF scores
    
    Use nltk library.
    """
    outputStr = inputStr
    return len(outputStr)
    
    
    
    
    
    