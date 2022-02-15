import sys
import tweepy as tw


def twitter_auth():

    try:
        consumer_key = "x"
        consumer_secret = "x"
        access_token = "x-x"
        access_secret = "x"
    except KeyError:
        sys.stderr.write("Key or secret token are invalid.\n")
        sys.exit(1)

    auth = tw.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)

    return auth


def twitter_client():

    auth = twitter_auth()
    client = tw.API(auth)

    return client
