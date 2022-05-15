#!/usr/bin/env python
from nltk import FreqDist
from nltk.corpus import treebank
from nltk import chunk

# import ne chunker
data = [item for data_list in treebank.tagged_sents() for item in data_list]
chunkd_data = chunk.ne_chunk(data, binary=True)
chunkd_trees = chunkd_data.subtrees(filter=lambda t: t.label() == "NE")

word_fd = FreqDist(
    [" ".join(word for word, _ in tree.leaves()) for tree in chunkd_trees]
)
print("Three most common named entities are: ")
for token, freq in word_fd.most_common(3):

    print("%s : %d" % (token, freq))
