import pandas
import random
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from random import sample, seed
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class DataLoader():
    def __init__(self, filename, size):
        self.df = pandas.read_csv(filename, low_memory=False, lineterminator='\n')
        seed(0)
        sid = SentimentIntensityAnalyzer()
        self.tweets = sample(self.df['text'].astype(str).tolist(), size)
        self.scores = [self.convert_polarity_score(sid.polarity_scores(x)) for x in self.tweets]
        print("Dataset size:", len(self.tweets))
    
    def load_dataset(self, return_X_y=True):
        if return_X_y==False:
            return self.df
        return self.tweets, self.scores
    
    def set_X(self, dataX, dataY):
        self.tweets = dataX
        self.scores = dataY

    def train_test_split(self, X, y, ratio):
        if X is None:
            X=self.tweets
        if y is None:
            y=self.scores
        zipped = list(zip(X, y))
        random.shuffle(zipped)
        tweets, scores = zip(*zipped)

        split_pos = int(len(tweets)*ratio)

        x_train = tweets[:split_pos]
        x_test = tweets[split_pos:]
        y_train = scores[:split_pos]
        y_test = scores[split_pos:]

        return x_train, x_test, y_train, y_test 

    def convert_polarity_score(self, scores):
        return scores['compound']

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


    
    
        
