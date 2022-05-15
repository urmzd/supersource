#!/usr/bin/env python
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords  # We imported auxiliary corpus

# provided with NLTK
data = """All work and no play makes jack dull boy.
        All work and no play makes jack a dull boy."""
stopWords = set(stopwords.words("english"))  # a set of English
words = word_tokenize(data.lower())  # stopwords
wordsFiltered = []

for w in words:
    if w not in stopWords:
        wordsFiltered.append(w)

print(len(stopWords))  # Print the number of stopwords

print(stopWords)  # Print the stopwords

print(wordsFiltered)  # Print the filtered text
