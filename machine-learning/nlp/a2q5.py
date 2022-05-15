#!/usr/bin/env python
# @author: Urmzd Mukhammadnaim
# @file: a2q5.py
# @class: CSCI 4152
# @banner_no: B00800045
# @purpose: Provides functionality to determine class-level precision, recall and F1-measure.
#           Additionally, this file contains utility helper to determine macro-averages.

from os import linesep
import re
from typing import List, Dict, Union

# Types.
Labels = List[str]
ConfusionMatrix = Dict[str, Dict[str, int]]


def read_input():
    true_labels = []
    predicted_labels = []
    stdin = []
    while True:
        _in = str(input())
        if "end" not in _in:
            pattern = r"true label:([^\s]) result:([^\s])"
            true, pred = re.findall(pattern, _in)[0]
            true_labels.append(true)
            predicted_labels.append(pred)

            stdin.append(_in)
        else:
            return (true_labels, predicted_labels)


def precision(tp: int, fp: int):
    return tp / (tp + fp) if tp or fp else 0


def recall(tp: int, fn: int):
    return tp / (tp + fn) if tp or fn else 0


def f_measure(p: float, r: float):
    return 2 * (p * r) / (p + r) if p or r else 0


def confusion_matrix(true_labels: Labels, predicted_labels: Labels) -> ConfusionMatrix:
    labels = sorted(list({*true_labels, *predicted_labels}))
    con_mat = {p: {t: 0 for t in labels} for p in labels}

    n_entries = len(true_labels)

    for i in range(n_entries):
        true_label = true_labels[i]
        predicted_label = predicted_labels[i]

        con_mat[predicted_label][true_label] += 1

    return con_mat


def class_precision(label: str, confusion_matrix: ConfusionMatrix) -> float:
    tp = confusion_matrix[label][label]
    fp = sum([v for k, v in confusion_matrix[label].items() if k != label])

    return precision(tp, fp)


def all_class_precision(confusion_matrix: ConfusionMatrix):
    return [class_precision(k, confusion_matrix) for k in confusion_matrix]


def class_recall(label: str, confusion_matrix: ConfusionMatrix) -> float:
    tp = confusion_matrix[label][label]
    fn = sum([v[label] for k, v in confusion_matrix.items() if k != label])

    return recall(tp, fn)


def all_class_recall(confusion_matrix: ConfusionMatrix):
    return [class_recall(k, confusion_matrix) for k in confusion_matrix]


def class_f_measure(label: str, confusion_matrix: ConfusionMatrix) -> float:
    return f_measure(
        class_precision(label, confusion_matrix), class_recall(label, confusion_matrix)
    )


def all_class_f_measure(confusion_matrix: ConfusionMatrix):
    return [class_f_measure(k, confusion_matrix) for k in confusion_matrix]


def macro_average(*measurements: List[Union[float, int]]):
    return sum(*measurements) / len(*measurements)


def pretty_stdout(labels, acp, acr, acf, mac_acp, mac_acr, mac_acf) -> None:
    for index, label in enumerate(labels):
        print(
            f"P({label})={acp[index]:.5f} R({label})={acr[index]:.5f} F1({label})={acf[index]:.5f}"
        )

    print(f"P={mac_acp:.5f} R={mac_acr:.5f} F1={mac_acf:.5f}")


if __name__ == "__main__":
    stdin = read_input()
    con_mat = confusion_matrix(*stdin)
    labels = con_mat.keys()
    acp = all_class_precision(con_mat)
    acr = all_class_recall(con_mat)
    acf = all_class_f_measure(con_mat)
    mac_acp = macro_average(acp)
    mac_acr = macro_average(acr)

    mac_acf = f_measure(mac_acp, mac_acr)

    pretty_stdout(labels, acp, acr, acf, mac_acp, mac_acr, mac_acf)
