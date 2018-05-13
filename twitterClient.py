import sys
import tweepy as tw


def twitter_auth():

    try:
        consumer_key = "q3jskpUCOy2h64TgLIJVk4lYG"
        consumer_secret = "ihKjoH66jRlTypi5oFSye5exKnNyxpbLJREd5kf5ivrqBq9YjA"
        access_token = "36773543-oNFUjf69C58EKhCBNoeWHUVTrHHAKnmtCCvJBeD9D"
        access_secret = "u8milltoVvrRPAEl05TC70PbCuFiV8FH0g2t1BkUFD0a5"
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
