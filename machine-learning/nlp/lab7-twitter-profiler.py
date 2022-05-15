#!/usr/bin/env python
# File: lab7-twitter-profiler.py
# Twitter Profiler app. This is a simple script to configure the Twitter API
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

s = get_profile("DalhousieU")
# print(s)
print("Name: " + s.name)
print("Location: " + s.location)
print("Description: " + s.description)
