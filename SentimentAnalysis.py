"""
@author Rahul Gupta
"""

import string
import re
import tweepy
import json
import twitterClient

import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import opinion_lexicon
from collections import Counter

from colorama import Fore, Style
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn')


class TwitterProcessing:
    def __init__(self, tokeniser, lStopwords):
        self.tokeniser = tokeniser
        self.lStopwords = lStopwords

    def process(self, text):
        text = text.lower()
        tokens = self.tokeniser.tokenize(text)
        tokensStripped = [tok.strip() for tok in tokens]
        regexDigit = re.compile("^\d+\s|\s\d+\s|\s\d+$")
        regexHttp = re.compile("^http")

        return [tok for tok in tokensStripped if tok not in self.lStopwords and
                regexDigit.match(tok) == None and regexHttp.match(tok) == None]


output_file = 'streaming_service.json'

with open(output_file, 'r') as f:
    for line in f:
        tweets = json.loads(line)

tweetTokenizer = TweetTokenizer()
lPunct = list(string.punctuation)
lStopwords = stopwords.words('english') + lPunct + ['rt', 'via', '...', '…', '"', "'", '`']

tweetProcessor = TwitterProcessing(tweetTokenizer, lStopwords)




def get_frequent_keywords(tweets, tweetProcessor):
    tweet_text = set([tweet['text'] for tweet in tweets])
    freq_counter = Counter()

    for tweet in tweet_text:
        tokens = tweetProcessor.process(tweet)
        freq_counter.update(tokens)
    return freq_counter


if __name__ == '__main__':
    get_frequent_keywords(tweets, tweetProcessor)
