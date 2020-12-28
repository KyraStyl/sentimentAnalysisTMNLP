#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 14:25:29 2020

@author: kyrastyl

This is a python script which works as a library.

Write here all preprocessing functions.

"""

import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

stopword_list = stopwords.words("english")
stemmer = WordNetLemmatizer()

def removeURLs(inputStr):
    """
    :param inputStr: the input string
    :return outputStr: inputStr without urls
    """
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', ' URL ', inputStr)
    return tweet
    
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
    tokens = word_tokenize(inputStr)
    tokens = [x for x in tokens if x not in stopword_list]
    outputStr = ' '.join(tokens)
    return outputStr

def lemmatize(inputStr):
    """
    :param inputStr: the input string
    :return outputStr: inputStr lemmatized

    Replace words with their lemmatization, i.e. 'am', 'is', 'are' => 'be' 
    """
    tokens = word_tokenize(inputStr)
    lemmatized = [stemmer.lemmatize(word) for word in tokens]
    outputStr = ' '.join(lemmatized)
    return outputStr

def removeNumbers(inputStr):
    """
    :param inputStr: the input string
    :return outputStr: inputStr with numbers replaced by 'number'

    The text "This is 20.13 metres long" becomes "This is number metres long".
    """
    outputStr = re.sub(r'(?<=\d), [,\.]', '', inputStr)
    outputStr = re.sub(" \d+", " number ", inputStr)
    return outputStr

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
    
    
    
    
    
    