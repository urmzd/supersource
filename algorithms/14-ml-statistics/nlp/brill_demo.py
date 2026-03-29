#!/usr/bin/env python

from nltk.tbl.demo import postag

postag(num_sents=None, train=0.7665)
# if we set num_sents to None, it will use the whole treebank corpus.
# We want this, so we can compare the results to the CRF and HMM we
# tested earlier. If we set train ratio to 0.7665, the train set will have
# 3000 sentences, just like in previous taggers. The other params are default.

# Accuracy: 0.9242 

# Top 3 Rules:
# NN->VB if Pos:-NONE-@[-2] & Pos:TO@[-1]
# VBP->VB if Pos:MD@[-3,-2,-1]
# VBP->VB if Pos:TO@[-1]

# It is better than HMM but not CRF
