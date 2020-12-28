from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from nltk.tokenize import word_tokenize

class W2V():
    def __init__(self):
        self.wv = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit=500000)

    def get_w2v(self, word):
        try:
            return self.wv[word]
        except:
            return 0

    def make_embedding(self, sentence):
        sentence_list = word_tokenize(sentence)
        w2v_sentence = np.array([self.get_w2v(word) for word in sentence_list])
        return w2v_sentence
        
w = W2V()
print(w.make_embedding("hello world"))


