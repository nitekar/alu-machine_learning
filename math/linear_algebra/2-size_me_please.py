#!/usr/bin/env python3
"""calculating the shape of a matrix using recursion"""


def matrix_shape(matrix):
    """Calculating the matrix's shape"""
    shape = []
    while type(matrix) == list:
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
