#!/usr/bin/env python
import numpy as np
import pandas as pd
from tabulate import tabulate
from typing import Union, Dict
from functools import reduce

# Utils
def mapr(x: bool) -> int:
    return 0 if x else 1


def product(*args: float) -> float:
    return reduce(lambda x, y: x * y, args)


def concat_sum(cpt: np.ndarray, axis, transpose=False):
    sums = np.array([np.sum(cpt, axis=axis) for _ in range(cpt.shape[axis] - 1)])

    if transpose:
        sums = sums.T

    return np.concatenate((cpt, sums), axis=axis)

# Fully Independent Tasks
def fi_evaluate(df: pd.DataFrame, c, d, e, f):
    total_articles = df["articles"].sum()
    d_p = df[df["D"] == d]["articles"].sum() / total_articles
    e_p = df[df["E"] == e]["articles"].sum() / total_articles
    f_p = df[df["F"] == f]["articles"].sum() / total_articles
    c_p = df[df["C"] == c]["articles"].sum() / total_articles
    return product(d_p, e_p, f_p, c_p)

def fi_conditioning(df: pd.DataFrame, c, d, e, f):
    return fi_evaluate(df, c, d, e, f)

# Joint Probability Tasks
def jp_evaluate(df: pd.DataFrame, d: bool, e: bool, f: bool, c: bool = None):
    total_articles = np.sum(df["articles"])
    articles = np.sum(
        df.iloc[
            df.index[
                (True if c == None else df["C"].eq(c))
                & df["D"].eq(d)
                & df["E"].eq(e)
                & df["F"].eq(f)
            ]
        ]["articles"]
    )
    return articles / total_articles


def jp_conditioning(df: pd.DataFrame, c: bool, d: bool, e: bool, f: bool):
    return jp_evaluate(df, d, e, f, c) / jp_evaluate(df, d, e, f)


# Naive Bayes Tasks.
def nb_evaluate(cpts: Dict[str, np.ndarray], c, d, e, f):
    c_prob = cpts["C"][mapr(c), 0]
    d_prob = cpts["D"][mapr(d), mapr(c)]
    e_prob = cpts["E"][mapr(e), mapr(c)]
    f_prob = cpts["F"][mapr(f), mapr(c)]

    return product(c_prob, d_prob, e_prob, f_prob)


def nb_completion(cpts: Dict[str, np.ndarray], d, e, f):
    return (
        True
        if nb_evaluate(cpts, True, d, e, f) > nb_evaluate(cpts, False, d, e, f)
        else False
    )


def nb_conditioning(cpts: Dict[str, np.ndarray], c: bool, d: bool, e: bool, f: bool):
    prob_c = nb_evaluate(cpts, c, d, e, f)
    prob_not_c = nb_evaluate(cpts, (not c), d, e, f)

    return prob_c / (prob_c + prob_not_c)


def nb_generate_cpt(
    df: pd.DataFrame, source: str, target: Union[str, None] = None, prob=True
):
    """
    Generate a CPT for P({source} | {target})

    :param df pd.DataFrame: The input containing the the relationships between the feature and the target.
    :param source str: A property of the dataframe.
    :param target str: A property of the dataframe.
    """

    x_axis = 0
    source_values = [True, False]
    total_articles = np.sum(df["articles"])

    if not target:
        cpt = np.zeros(shape=(len(source_values), 1))
        print(f"CPT for P({source})")
        for i, f_v in np.ndenumerate(source_values):
            no_of_articles = df.loc[df.index[df[source] == f_v], "articles"].sum()
            cpt[i, 0] = no_of_articles

        if prob:
            cpt /= total_articles

        print(
            tabulate(
                cpt,
                headers=[f"P({source})"],
                showindex=[f"{source} = True", f"{source} = False"],
                tablefmt="pretty",
            )
        )
        print("\n")

    else:
        target_values = df[target].unique()

        cpt = np.zeros(shape=(len(source_values), len(target_values)))

        print(f"CPT for P({source}|{target})")
        for i, f_v in np.ndenumerate(source_values):
            for j, t_v in np.ndenumerate(source_values):
                # Sum the no. of articles where the feature is f_v and the target is t_v
                no_of_articles = df.loc[
                    df.index[(df[source] == f_v) & (df[target] == t_v)], "articles"
                ].sum()

                cpt[i, j] = no_of_articles

        # Column sums.
        cpt = concat_sum(cpt, x_axis)

        # Convert columns to probabilities.
        if prob:
            cpt[:2,] /= cpt[
                -1,
            ]

        cpt = cpt[
            :2,
        ]

        print(
            tabulate(
                cpt,
                headers=[
                    f"{target} = True",
                    f"{target} = False",
                ],
                showindex=[
                    f"{source} = True",
                    f"{source} = False",
                ],
                tablefmt="pretty",
            )
        )
        print("\n")

    return cpt

if __name__ == "__main__":
    df = pd.read_csv("a3q5.in", sep=" ")
    if isinstance(df, pd.DataFrame):
        df.replace({"f": False, "t": True}, inplace=True)
        no_articles = np.sum(df["articles"])

        difference = df.columns.difference(["articles", "C"])

        cpts = dict()

        if isinstance(difference, pd.Index):
            print("a)\n")
            cpts["C"] = nb_generate_cpt(df, "C", None)

            for feature in difference:
                cpts[f"{feature}"] = nb_generate_cpt(df, str(feature), "C")

            print("b)")
            print(nb_conditioning(cpts, True, False, True, True))
            print()

            print("c)")
            print(nb_completion(cpts, False, True, True))
            print()

            print("d)")
            print(jp_conditioning(df, True, False, True, True))
            print()

            print("e)")
            print(fi_conditioning(df, True, False, True, True))
            print()
