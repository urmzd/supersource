#!/usr/bin/env python
from nltk import FreqDist, NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.classify import accuracy
import random

documents = [(list(movie_reviews.words(fileid)), category)
for category in movie_reviews.categories()
for fileid in movie_reviews.fileids(category)]

random.shuffle(documents) # This line shuffles the order of the documents

all_words = FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words)[:2000]

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

featuresets = [(document_features(d), c) for (d,c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100] # Split

# data to train and test set
classifier = NaiveBayesClassifier.train(train_set)
print(accuracy(classifier, test_set))

# 0.75
# 0.74
# 0.8
# 0.79
# 0.83

# We accuracy differs between each iteration due to the shuffle that occurs in the beginning.
# By doing so, the training and testing sets differ. Thus, producing different result. 
