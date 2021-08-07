import nltk
import re

nltk.download(['punkt', 'wordnet', 'stopwords'])

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def tokenize(text):
    # get list of urls
    detected_urls = re.findall(url_regex, text)

    # replace url in text with urlplaceholder
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')

    text = re.sub(r'[^a-zA-Z0-9]', ' ', text).lower()
    tokens = word_tokenize(text)

    # Remove stop words
    words = [w for w in tokens if w not in stopwords.words("english")]

    # Lemmatize verbs by specifying pos
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in words]

    # Reduce words in lemmed to their stems
    stem_words = [PorterStemmer().stem(w) for w in lemmed]

    return stem_words


class DisResExtractor(BaseEstimator, TransformerMixin):

    def DisResWord(self, text):

        dis_res_words = ['drink',
                         'eat',
                         'thirst',
                         'medicine',
                         'cold',
                         'earthquake',
                         'floods',
                         'fire',
                         'storm',
                         'military',
                         'security',
                         'horrible',
                         'medical',
                         'cloth',
                         'shelter',
                         'hunger',
                         'hungry',
                         'food',
                         'water',
                         'help',
                         ]

        lemmed_dis_res_words = [WordNetLemmatizer().lemmatize(w, pos='v') for w in dis_res_words]
        stem_dis_res_words = [PorterStemmer().stem(w) for w in lemmed_dis_res_words]

        stem_words = tokenize(text)

        # return whether stem_words contains any of words in stem_dis_words
        return any([words in stem_dis_res_words for words in stem_words])

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.Series(X).apply(self.DisResWord)
        return pd.DataFrame(X)
