# -*- coding: utf-8 -*-
from numpy import allclose, array_equal, asanyarray, cross, isclose
from numpy.linalg import norm

from .config import config


def are_parallel(a, b):
    a_array = asanyarray(a)
    b_array = asanyarray(b)
    if not array_equal(a_array == 0, b_array == 0):
        return False
    return isclose(
        norm(cross(a_array, b_array)),
        0,
        **config['numbers_close_kwargs'])


def are_antiparallel(a, b):
    a_array = asanyarray(a)
    b_array = asanyarray(b)
    if (
            allclose(a_array, 0, **config['numbers_close_kwargs']) or
            allclose(b_array, 0, **config['numbers_close_kwargs'])):
        return False
    if allclose(
            a_array/norm(a_array),
            -b_array/norm(b_array),
            **config['numbers_close_kwargs']):
        return True
    return False
