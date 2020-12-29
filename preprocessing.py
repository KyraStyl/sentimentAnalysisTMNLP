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
import nltk
from nltk.tokenize import TweetTokenizer # ??
import string
import itertools
import emoji

nltk.download
nltk.download('stopwords')


stopwordsEN = stopwords.words("english")
stemmer = WordNetLemmatizer()
bad_chars = ['#','@',',','=','.','«','»','…',' ', "-","_", ":", ";", "(", ")", "{", "}","[","]","&","'", "!", "?","~", "\""]
             
             
def preprocess(inputStr):
    """
    Basic function for preprocessing.
    In this function is implemented the pipeline 
    for preprocessing each tweet.
    """
    outputStr = inputStr
    # find the appropriate pipeline to clean the data,
    # in such way that to not destroy them ;P
    
    # LAST STEP !
    return outputStr

def dealWithHashtags(inputStr):
    """
    :param inputStr: the input string
    :return outputStr: inputStr without hashtags
    
    Maybe remove the special character and keep only the word 
    from hashtag as a new "feature"
    """
    outputStr = inputStr
    return len(outputStr)

def dealWithContractions(inputStr):
    """
    :param inputStr: the input string
    :return outputStr: inputStr without contractions
    
    Use the contraction's dictionary to deal with contractions.
    
    e.g. didn't -> did not
    """
    outputStr = inputStr.replace("’","'")
    words = outputStr.split()
    CONTRACTIONS = load_dict_contractions()
    withoutContractions = [CONTRACTIONS[word] if word in CONTRACTIONS else word for word in words]
    outputStr = " ".join(withoutContractions)
    return outputStr
             

# additions need to be done
def removeHTMLandURLs(inputStr):
    """
    :param inputStr: the input string
    :return outputStr: inputStr without urls and HTML tags if present
    
    Hint: Use Beautiful soup library for HTML and re library for urls
    """
    outputStr = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', ' URL ', inputStr)
    return outputStr
    
def removeSpecialChars(inputStr):
    """
    :param inputStr: the input string
    :return outputStr: inputStr without special characters
    
    Remove special characters from input string.
    Special characters: #,@,!,-, etc.
    Also remove digits!
    """
    #replace with ' ' or '' ??
    #replace all "newline" tokens? -> \n ?
    outputStr = inputStr
    for i in bad_chars:
        outputStr = outputStr.replace(i, '')
    for i in string.punctuation : 
        outputStr = outputStr.replace(i, '')
    outputStr = re.sub(r'\d', '', outputStr)
    outputStr = re.sub(r'[^\w]', '',outputStr)
    return outputStr

def removeStopWords(inputStr):
    """
    :param inputStr: the input string
    :return outputStr: inputStr without stopwords
    
    Remove all stopwords using library nltk
    (Stopwords: "a","the","and", etc.)
    """
    tokens = word_tokenize(inputStr)
    tokens = [w for w in tokens if (len(w)>1)]
    tokens = [x for x in tokens if x not in stopwordsEN]
    outputStr = ' '.join(tokens)
    return outputStr

def fixMisspells(inputStr):
    """
    :param inputStr: the input string
    :return outputStr: inputStr without misspells
    
    Naive technique to fix misspelled words. 
    Each character should occur not more than 1 times consecutive in every word.
    
    e.g. goaaaaaal -> goal
    """
    outputStr = ''.join(''.join(s)[:1] for _, s in itertools.groupby(inputStr))
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
    SMILEY = load_dict_smileys()
    words = inputStr.split()
    withoutSmileys = [SMILEY[word] if word in SMILEY else word for word in words]
    outputStr = " ".join(withoutSmileys)
    outputStr = emoji.demojize(outputStr)
    outputStr = re.sub(r'\:','',outputStr)
    return outputStr

def dealWithRTsAndMentions(inputStr):
    """
    :param inputStr: the input string
    :return outputStr: inputStr without re-tweets or other user mentions
    
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
    
    
def load_dict_smileys():
    
    return {
        ":‑)":"happy",
        ":-]":"happy",
        ":-3":"happy",
        ":->":"happy",
        "8-)":"happy",
        ":-}":"happy",
        ":)":"happy",
        ":]":"happy",
        ":3":"happy",
        ":>":"happy",
        "8)":"happy",
        ":}":"happy",
        ":o)":"happy",
        ":c)":"happy",
        ":^)":"happy",
        "=]":"happy",
        "=)":"happy",
        ":-))":"happy",
        ":‑D":"happy",
        "8‑D":"happy",
        "x‑D":"happy",
        "X‑D":"happy",
        ":D":"happy",
        "8D":"happy",
        "xD":"happy",
        "XD":"happy",
        ":‑(":"sad",
        ":‑c":"sad",
        ":‑<":"sad",
        ":‑[":"sad",
        ":(":"sad",
        ":c":"sad",
        ":<":"sad",
        ":[":"sad",
        ":-||":"sad",
        ">:[":"sad",
        ":{":"sad",
        ":@":"sad",
        ">:(":"sad",
        ":'‑(":"sad",
        ":'(":"sad",
        ":‑P":"playful",
        "X‑P":"playful",
        "x‑p":"playful",
        ":‑p":"playful",
        ":‑Þ":"playful",
        ":‑þ":"playful",
        ":‑b":"playful",
        ":P":"playful",
        "XP":"playful",
        "xp":"playful",
        ":p":"playful",
        ":Þ":"playful",
        ":þ":"playful",
        ":b":"playful",
        "<3":"love"
        }

def load_dict_contractions():
    
    return {
        "ain't":"is not",
        "amn't":"am not",
        "aren't":"are not",
        "can't":"cannot",
        "'cause":"because",
        "couldn't":"could not",
        "couldn't've":"could not have",
        "could've":"could have",
        "daren't":"dare not",
        "daresn't":"dare not",
        "dasn't":"dare not",
        "didn't":"did not",
        "doesn't":"does not",
        "don't":"do not",
        "dunno":"do not know",
        "e'er":"ever",
        "em":"them",
        "everyone's":"everyone is",
        "finna":"fixing to",
        "gimme":"give me",
        "gonna":"going to",
        "gon't":"go not",
        "gotta":"got to",
        "hadn't":"had not",
        "hasn't":"has not",
        "haven't":"have not",
        "he'd":"he would",
        "he'll":"he will",
        "he's":"he is",
        "he've":"he have",
        "how'd":"how would",
        "how'll":"how will",
        "how're":"how are",
        "how's":"how is",
        "I'd":"I would",
        "I'll":"I will",
        "I'm":"I am",
        "I'm'a":"I am about to",
        "I'm'o":"I am going to",
        "isn't":"is not",
        "it'd":"it would",
        "it'll":"it will",
        "it's":"it is",
        "I've":"I have",
        "kinda":"kind of",
        "lemme":"let me",
        "let's":"let us",
        "mayn't":"may not",
        "may've":"may have",
        "mightn't":"might not",
        "might've":"might have",
        "mustn't":"must not",
        "mustn't've":"must not have",
        "must've":"must have",
        "needn't":"need not",
        "ne'er":"never",
        "o'":"of",
        "o'er":"over",
        "ol'":"old",
        "oughtn't":"ought not",
        "shalln't":"shall not",
        "shan't":"shall not",
        "she'd":"she would",
        "she'll":"she will",
        "she's":"she is",
        "shouldn't":"should not",
        "shouldn't've":"should not have",
        "should've":"should have",
        "somebody's":"somebody is",
        "someone's":"someone is",
        "something's":"something is",
        "that'd":"that would",
        "that'll":"that will",
        "that're":"that are",
        "that's":"that is",
        "there'd":"there would",
        "there'll":"there will",
        "there're":"there are",
        "there's":"there is",
        "these're":"these are",
        "they'd":"they would",
        "they'll":"they will",
        "they're":"they are",
        "they've":"they have",
        "this's":"this is",
        "those're":"those are",
        "'tis":"it is",
        "'twas":"it was",
        "wanna":"want to",
        "wasn't":"was not",
        "whatcha":"what are you",
        "we'd":"we would",
        "we'd've":"we would have",
        "we'll":"we will",
        "we're":"we are",
        "weren't":"were not",
        "we've":"we have",
        "what'd":"what did",
        "what'll":"what will",
        "what're":"what are",
        "what's":"what is",
        "what've":"what have",
        "when's":"when is",
        "where'd":"where did",
        "where're":"where are",
        "where's":"where is",
        "where've":"where have",
        "which's":"which is",
        "who'd":"who would",
        "who'd've":"who would have",
        "who'll":"who will",
        "who're":"who are",
        "who's":"who is",
        "who've":"who have",
        "why'd":"why did",
        "why're":"why are",
        "why's":"why is",
        "won't":"will not",
        "wouldn't":"would not",
        "would've":"would have",
        "ya":"you",
        "y'all":"you all",
        "you'd":"you would",
        "you'll":"you will",
        "you're":"you are",
        "you've":"you have",
        "Whatcha":"What are you",
        "luv":"love",
        "sux":"sucks"
        }
    
    
    
