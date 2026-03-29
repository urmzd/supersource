#!/usr/bin/env python
# @author Urmzd Mukhammadnaim
# @date 2021/11/12
# @file a3q6.py
# @purpose Calculate the edit distance of pairs given from stdin.
# @class CSCI4152 
# @assignment Assignment #3 Quesiton #6
# @banner B00800045
# @csid urmzd

# Regex Library
import re

def calculate_edit_distance(S: str, T: str) -> int:
    """
    Calculate the Edit Distance between S & T using DP

    :param S str: The source string.
    :param T str: The target string.
    :rtype int: The cost of changing S to T.
    """
    M, N = len(S) + 1, len(T) + 1

    EDM = [list(range(N)) if not i else [i] + [0] * (N - 1)
         for i in range(M)]

    for i in range(1, M):
        for j in range(1, N):
            match_cost = S[i - 1] != T[j - 1]

            change_cost = EDM[i-1][j-1] + match_cost
            insert_cost = EDM[i-1][j] + 1
            delete_cost = EDM[i][j-1] + 1

            EDM[i][j] = min(change_cost, insert_cost, delete_cost)

    return EDM[-1][-1]

if __name__ == '__main__':
    edit_distances = []
    while(True):
        source = (input()).strip()

        if (re.match(f"END", source)):
            break;

        target = (input()).strip()

        if (re.match(f"END", target)):
            break;
        else:
            edit_distances.append(calculate_edit_distance(source, target))

    for edit_distance in edit_distances:
        print(edit_distance)
