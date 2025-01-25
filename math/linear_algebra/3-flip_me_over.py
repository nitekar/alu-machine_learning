#!/usr/bin/env python3
"""Creates a func to return a matrix transpose"""


def matrix_transpose(matrix):
    """Returns a matrix transpose"""
    transpose = [[matrix[j][i] for j in range(len(matrix))]
                 for i in range(len(matrix[0]))]
    return transpose
