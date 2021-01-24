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
            return [0.0]*300

    def make_embedding(self, sentence):
        sentence_list = word_tokenize(sentence)
        if len(sentence_list)==0:
            return np.array([0]*300)
        w2v_sentence = [self.get_w2v(word) for word in sentence_list]
        w2v_sentence = np.array(w2v_sentence)
        return np.mean(w2v_sentence, axis=0)

#print(W2V().make_embedding('future grandson hey grandpa want corona'))