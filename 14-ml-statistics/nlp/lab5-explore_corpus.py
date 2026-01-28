#!/usr/bin/env python
from nltk.corpus import gutenberg
from nltk import FreqDist

# Count each token in austen-persuasion.txt of the Gutenberg collection
list_of_words = gutenberg.words("austen-persuasion.txt")
fd = FreqDist(list_of_words) # Frequency distribution object
print("Total number of tokens: " + str(fd.N())) # 98171
print("Number of unique tokens: " + str(fd.B())) # 6132
print("Top 10 tokens:") # to 
for token, freq in fd.most_common(10):
    print(token + "\t" + str(freq))

