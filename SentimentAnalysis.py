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

from colorama import Fore, Style
import pandas as pd
import matplotlib.pyplot as plt

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


def computeSentiment(lTokens, setPosWords, setNegWords):
    posNum = len([tok for tok in lTokens if tok in setPosWords])
    negNum = len([tok for tok in lTokens if tok in setNegWords])

    sentimentVal = posNum - negNum
    return sentimentVal


def countWordSentimentAnalysis(setPosWords, setNegWords, tweets, tweetProcessor):
    lSentiment = []
    for tweet in tweets:
        try:
            tweetText = tweet['text']
            tweetDate = tweet['created_at']

            lTokens = tweetProcessor.process(tweetText)
            sentiment = computeSentiment(lTokens, setPosWords, setNegWords)
            lSentiment.append([pd.to_datetime(tweetDate), tweetText, sentiment])
        except KeyError as e:
            pass

    return lSentiment


def vaderSentimentAnalysis(tweets, bPrint, tweetProcessor):
    sentAnalyser = SentimentIntensityAnalyzer()

    lSentiment = []

    for tweet in tweets:
        try:
            tweetText = tweet['text']
            tweetDate = tweet['created_at']

            lTokens = tweetProcessor.process(tweetText)

            dSentimentScores = sentAnalyser.polarity_scores(" ".join(lTokens))

            lSentiment.append([pd.to_datetime(tweetDate), tweetText, dSentimentScores['compound']])

            if bPrint:
                print(*lTokens, sep=', ')
                for cat, score in dSentimentScores.items():
                    print('{0}: {1}, '.format(cat, score), end='')
                print()

        except KeyError as e:
            pass

    return lSentiment

def main():
    api = twitterClient.twitter_client()

    tweets = []
    query = 'streaming service OR streaming movies OR streaming series'
    # query = 'happy'
    max_tweets = 500

    # append all tweet data to list
    file_name = 'sentiment_rawdata.json'

    for tweet in tweepy.Cursor(api.search, q=query, lang="en").items(max_tweets):
        tweets.append(tweet._json)

    with open(file_name, 'w') as f:
        json.dump(tweets, f)

    with open(file_name, 'r') as f:
        for line in f:
            tweets = json.loads(line)

    tweetTokenizer = TweetTokenizer()
    lPunct = list(string.punctuation)
    lStopwords = stopwords.words('english') + lPunct + ['rt', 'via', '...', '…', '"', "'", '`']

    tweetProcessor = TwitterProcessing(tweetTokenizer, lStopwords)

    PosWords = opinion_lexicon.positive()
    NegWords = opinion_lexicon.negative()

    lSentiment = countWordSentimentAnalysis(PosWords, NegWords, tweets, tweetProcessor)
    print(lSentiment)
    print(len(lSentiment))

    df = pd.DataFrame(lSentiment)
    df.columns = ['Timestamp', 'text','Sentiment']

    freq = df.groupby(['Sentiment']).size().reset_index(name='counts')
    freq.sort_values(['counts'], ascending=False, inplace=True)
    freq = freq.head(20)

    freq.plot(kind='bar', x='Sentiment', y='counts', color='#006699')
    ax = plt.gca().invert_yaxis()
    plt.xlabel('Word Count')
    plt.ylabel('Words')
    plt.title('Bar chart of word frequency')
    plt.legend().set_visible(False)
    plt.show()



if __name__ == '__main__':
    main()