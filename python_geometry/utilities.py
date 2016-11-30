# -*- coding: utf-8 -*-
from numpy import allclose, asanyarray, cross, mean, zeros
from numpy.linalg import norm


def get_normal_vector(points):
    points = asanyarray(points)
    if points.ndim == 2 and len(points) <= 2:
        return zeros(points.shape[1])
    segments = points[1:] - points[:-1]
    normal_vector = mean(cross(segments[:-1], segments[1:], axis=1), axis=0)
    if allclose(normal_vector, 0):
        return zeros(points.shape[1])
    return normal_vector/norm(normal_vector)
