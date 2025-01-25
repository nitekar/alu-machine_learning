#!/usr/bin/env python3
"""Creates a funct to adds two arrays element-wise"""


def add_arrays(arr1, arr2):
    """Adding two arrays element-wise"""
    if len(arr1) != len(arr2):
        return None
    return [arr1[i] + arr2[i] for i in range(len(arr1))]
