#!/usr/bin/env python
# File: lab7-tweets.py
# Retrieving user tweets 
import tweepy # https://github.com/tweepy/tweepy
import time

consumer_key = "FC1HCyMBei**********"
# consumer_secret is API Key Secret
consumer_secret = "lLXBeSYowLHHDCJS***********"
# access_key is Access Token
access_key = "2841901529-MAFxKe**********************"
# access_secret is Access Token Secret
access_secret = "6QcxzJR**********************"
# this function collects a twitter profile request and returns a
# Twitter object
def get_profile(screen_name):
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)
    try:
        user_profile = api.get_user(screen_name=screen_name)
    except:
        user_profile = "broken"
    return user_profile

# this function collects twitter profile tweets and returns a Tweet
# object
def get_tweets(screen_name):
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)
    try:
        # https://developer.twitter.com/en/docs/tweets/timelines/overview
        # describes user_timeline
        tweets = api.user_timeline(screen_name=screen_name, count=20)
    except:
        tweets = "broken"
    return tweets

t = get_tweets("DalhousieU")
#print(t)
for tweet in t:
    print(tweet._json["text"] + "\n")
