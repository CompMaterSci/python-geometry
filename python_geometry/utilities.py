# -*- coding: utf-8 -*-
from __future__ import division

from numpy import allclose, asanyarray, cross, dot, mean, zeros
from numpy.linalg import norm

from .bound_vector import BoundVector
from .plane import Plane


def get_normal_vector(points):
    points = asanyarray(points)
    if points.ndim == 2 and len(points) <= 2:
        return zeros(points.shape[1])
    segments = points[1:] - points[:-1]
    normal_vector = mean(cross(segments[:-1], segments[1:], axis=1), axis=0)
    if allclose(normal_vector, 0):
        return zeros(points.shape[1])
    return normal_vector/norm(normal_vector)


def get_intersection(object_0, object_1):
    if isinstance(object_0, BoundVector) and isinstance(object_1, Plane):
        intersection = _get_intersection_bound_vector_plane(
            bound_vector=object_0,
            plane=object_1)
    elif isinstance(object_0, Plane) and isinstance(object_1, BoundVector):
        intersection = _get_intersection_bound_vector_plane(
            bound_vector=object_1,
            plane=object_0)
    else:
        raise NotImplementedError(
            'Intersection of a {} and a {} is not yet implemented.'.format(
                object_0.__class__.__name__,
                object_1.__class__.__name__))
    return intersection


def _get_intersection_bound_vector_plane(bound_vector, plane):

    param = (
        dot(
            plane.point_in_plane - bound_vector.initial_point,
            plane.normal_vector) /
        dot(
            bound_vector.terminal_point - bound_vector.initial_point,
            plane.normal_vector))
    if 0 <= param <= 1:
        intersection = (
            bound_vector.initial_point +
            param*(bound_vector.terminal_point - bound_vector.initial_point))
    else:
        intersection = None
    return intersection
