import pandas
import random
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

class DataLoader():
    def __init__(self, filename):
        self.df = pandas.read_csv(filename)
        self.tweets = self.df['text'].tolist()
        self.scores = self.df['sentiment_score'].tolist()  
    
    def load_dataset(self, return_X_y=True):
        if return_X_y==False:
            return self.df
        return self.tweets, self.scores
    
    def set_X(self, dataX):
        self.tweets = dataX

    def train_test_split(self, ratio):
        zipped = list(zip(self.tweets, self.scores))
        random.shuffle(zipped)
        tweets, scores = zip(*zipped)

        split_pos = int(len(tweets)*ratio)

        return tweets[:split_pos], tweets[split_pos:], scores[:split_pos], scores[split_pos:]

class ClassificationDataloader():
    def __init__(self, filename):
        df = pandas.read_csv(filename)
        tweets = df['text'].tolist()

        class1 = ['frustrated', 'distressed', 'annoyed', 'afraid', 'angry', 'tense', 'alarmed']
        class2 = ['miserable', 'sad', 'gloomy', 'depressed', 'bored', 'droopy', 'tired']
        class3 = ['sleepy', 'calm', 'relaxed', 'satisfied', 'atease', 'content', 'serene', 'glad', 'pleased']
        class4 = ['aroused', 'astonished', 'excited', 'delighted', 'happy']

        self.class1 = self.synonyms_of_class(class1)
        self.class2 = self.synonyms_of_class(class2)
        self.class3 = self.synonyms_of_class(class3)
        self.class4 = self.synonyms_of_class(class4)

        self.tweets = []
        self.labels = []

        for t in tweets:
            classes = self.check_sentence(t)
            if len(classes)==1:
                self.tweets.append(t)
                self.labels.append(list(classes)[0])

    
    def check_sentence(self, sentence):
        sentence = word_tokenize(sentence)
        sentence_classes = set()
        for w in sentence:
            if w in self.class1:
                sentence_classes.add(0)
            elif w in self.class2:
                sentence_classes.add(1)
            elif w in self.class3:
                sentence_classes.add(2)
            elif w in self.class4:
                sentence_classes.add(3)
        
        return sentence_classes
    
    def synonyms_of_class(self, sentiment_list):
        result = []
        for sentiment in sentiment_list:
            syn = wordnet.synsets(sentiment)
            l = [x.name().split('.')[0] for x in syn]
            result.extend(l)
        result = set(result)
        return result


    
    
        
