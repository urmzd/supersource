#!/usr/bin/env python
# File: lab7-hashtags.py
# Find tweet hashtags and tokens.

import csv
import nltk
import re

word_regex = re.compile(r"^[a-zA-Z]*$")
hashtag_regex = re.compile(r"^#.*$")

unique = dict()
hashtags = dict()
with open("lab7-tweets.csv", "r") as infile:
    reader = csv.reader(infile, quotechar='"')
    for row in reader:
        words = set(nltk.tokenize.TweetTokenizer().tokenize(row[-1]))
        unique[row[1]] = unique.get(row[1], set()).union(
            set(filter(word_regex.search, words))
        )
        hashtags[row[1]] = hashtags.get(row[1], set()).union(
            set(filter(hashtag_regex.search, words))
        )

    for k in unique:
        print(f"{k}:")
        print(f'unique tokens: {",".join(unique[k])}')
        print(f'hashtags: {",".join(hashtags[k])}')
