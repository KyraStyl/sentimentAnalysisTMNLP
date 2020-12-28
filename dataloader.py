import pandas
import random

class DataLoader():
    def __init__(self, filename):
        self.df = pandas.read_csv(filename)
        self.tweets = self.df['text'].tolist()
        self.scores = self.df['sentiment_score'].tolist()    

    def train_test_split(self, ratio):
        zipped = list(zip(self.tweets, self.scores))
        random.shuffle(zipped)
        tweets, scores = zip(*zipped)

        split_pos = int(len(tweets)*ratio)

        return tweets[:split_pos], tweets[split_pos:], scores[:split_pos], scores[split_pos:]


    