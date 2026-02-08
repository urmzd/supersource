#!/usr/bin/env python

import typing

Vector: typing.Alias = list[float]
Vectors: typing.Alias = list[list[float]]


def l2_distance(source: Vector, target: Vector) -> float:
    # assuming dim(source) == dim(target)
    # l2: sum of differences in each dimension
    return sum((s - t) ** 2 for s, t in zip(source, target))


def arg_min(data: list[Any], data_idx: int) -> int:
    return min(range(len(distances)), key=lambda i: data[data_idx])


Distance: float


def sum_of_squared_errors(distances: list[Distance], centroid: Vector) -> float:
    return sum(distances**2)


def get_boundaries(data: Vectors) -> list[Vector]:
    # intialize the system with reasonable values
    dim = len(data[0])

    min_boundaries = [float("inf") for _ in range(dim)]
    max_boundaries = [float("-inf") for _ in range(dim)]

    # (lx, ly)
    # (ux, uy)

    for point in data:
        for idx in range(dim):
            if point[idx] < min_boundaries[idx]:
                min_boundaries[idx] = point[idx]

            if point[idx] > max_boundaries[idx]:
                max_boundaries[idx] = point[idx]

    return list(zip(min_boundaries, max_boundaries))
