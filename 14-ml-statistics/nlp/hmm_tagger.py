#!/usr/bin/env python
# Import the toolkit and tags
from nltk.corpus import treebank

# Import HMM module
from nltk.tag import hmm

# Train data - pretagged
train_data = treebank.tagged_sents()[:3000]
# Test data - pretagged
test_data = treebank.tagged_sents()[3000:]
print(train_data[0])
# Setup a trainer with default(None) values
# And train with the data
trainer = hmm.HiddenMarkovModelTrainer()
tagger = trainer.train_supervised(train_data)
print(tagger)
# Prints the basic data about the tagger
print(tagger.tag("Today is a good day.".split()))
print(tagger.tag("Joe met Joanne in New Delhi.".split()))
print(tagger.tag("Chicago is the birthplace of Marry".split()))
# 'birthplace' is NNP, should be NN
# 'of' is NNP, should be IN
print(tagger.evaluate(test_data))
