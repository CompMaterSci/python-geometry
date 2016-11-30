# -*- coding: utf-8 -*-
from numpy import array_equal, asanyarray, cross, isclose
from numpy.linalg import norm


def are_parallel(a, b):
    a_array = asanyarray(a)
    b_array = asanyarray(b)
    if not array_equal(a_array == 0, b_array == 0):
        return False
    return isclose(norm(cross(a_array, b_array)), 0)
