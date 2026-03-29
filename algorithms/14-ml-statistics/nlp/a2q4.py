#!/usr/bin/python
import matplotlib.pyplot as plt
from typing import List, Tuple, Union
from tabulate import tabulate

results = [1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0]
truth = [1] * 25

def precision(tp: int, fp: int) -> float:
    """
    Calculate precision given the number of true and false positive results in a model.

    :param tp int: The number of items that were predicted correctly.
    :param fp int:  The number of items that were not relevant but predicted to be.
    :rtype float: The precision metric of the model.
    """
    return tp / (tp + fp)


def recall(tp: int, fn: int) -> float:
    """
    Calculate the recall given the number of true positives and the number of false negatives in a model.

    :param tp int: The number of items that were predicted to be relevant and are.
    :param fn int: The number of items that were predicted to be irrelevant and are not.
    :rtype float: The recall metric of the model.
    """
    return tp / (tp + fn)


def precision_recall(
    pred: List[int], truth: List[int], interpolate: bool = False
) -> Tuple[List[float], List[float]]:
    """
    Retrieve the relationship between precision and recall as the number of inputs increase.

    :param pred List[int]: The predicted values of the model.
    :param truth List[int]: The actual values of the model.
    :param interpolate bool: Optional parameter to determine whether to interpolate precision.
    """
    n_results = len(pred)

    all_precision = [
        precision(pred[:i].count(1), pred[:i].count(0)) for i in range(1, n_results + 1)
    ]

    all_recall = [
        recall(pred[:i].count(1), len(truth)) for i in range(1, n_results + 1)
    ]

    interpolated_precision: Union[List[float], None] = (
        [max(pred[i:]) for i in range(n_results)] if interpolate else None
    )

    precision_recall = [
        (
            interpolated_precision[i] if interpolated_precision else all_precision[i],
            all_recall[i],
        )
        for i in range(len(pred))
    ]

    print(
        tabulate(
            precision_recall,
            headers=[
                "Precision" if not interpolate else "Interpolated Precision",
                "Recall",
            ],
        )
    )

    return (
        interpolated_precision if interpolated_precision else all_precision,
        all_recall,
    )


pc_0 = precision_recall(results, truth)
pc_1 = precision_recall(results, truth, True)


def plot_precision_recall_curve(
    precision_recall: Tuple[List[float], List[float]], interpolated=False
) -> None:
    plt.close()
    precision_values = precision_recall[0]
    recall_values = precision_recall[1]
    plt.plot(recall_values, precision_values)
    plt.xlabel("Recall")
    plt.ylabel("Precision" if not interpolated else "Interpolated Precision")
    plt.savefig("images/precision_recall_curve.png" if not interpolated else "images/interpolated_precison_recall_curve.png")

plot_precision_recall_curve(pc_0)
plot_precision_recall_curve(pc_1)
