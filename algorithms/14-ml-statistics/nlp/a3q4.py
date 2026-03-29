#!/usr/bin/env python
import string
from tabulate import tabulate
from typing import Optional, TypedDict, Callable

phrase = "to be or not to be"
phrase_len = len(phrase)
vocabulary = [*string.ascii_lowercase, " "]
voc_len = len(vocabulary)


class ProbDist(TypedDict):
    character: list[str]
    probability: list[float]


def n_gram(smoothing: Callable[[float, str], float], label="Smoother"):

    prob_dist: ProbDist = {
        "character": [*vocabulary],
        "probability": [0.0 for _ in range(voc_len)],
    }

    for k in phrase:
        c = prob_dist["character"]
        index = c.index(k)
        prob_dist["probability"][index] += 1

    for k in prob_dist["character"]:
        c = prob_dist["character"]
        index = c.index(k)
        prob_dist["probability"][index] = smoothing(prob_dist["probability"][index], k)

    print(label)
    print(tabulate(prob_dist, headers="keys"))
    print()


## Question 1 - Unigram (No Smoothing)

n_gram(lambda x, _: x / phrase_len, "a) No Smoothing")

## Question 2 - Unigram (Laplace Add One Smoothing)

n_gram(lambda x, _: (x + 1) / (voc_len + phrase_len), "b) Laplace Add One Smoothing")

## Question 3 - Unigram (Witten-Bell Smoothing)

def witten_bell():
    unique_tokens = set(phrase)
    unique_tokens_len = len(unique_tokens)
    counts = {k: phrase.count(k) for k in unique_tokens}
    tot_len = unique_tokens_len + sum([count for _, count in counts.items()])
    unseen_event_prob = (unique_tokens_len) / (tot_len * (voc_len - unique_tokens_len))

    def smoother(_: float, k: str) -> float:
        if k in unique_tokens:
            return counts[k] / tot_len

        return unseen_event_prob

    return smoother


smoother_lambda = witten_bell()
n_gram(lambda x, k: smoother_lambda(x, k), "c) Witten-Bell Smoothing")
