# -*- coding: utf-8 -*-
from __future__ import division

from numpy import (
    allclose, asanyarray, cross, dot, errstate, isclose, mean, nan_to_num,
    nanmean, stack, zeros)
from numpy.linalg import det, norm

from .config import config
from .bound_vector import BoundVector
from .line_segment import LineSegment
from .plane import Plane
from .vector_utilities import are_antiparallel, are_parallel


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
    if isinstance(object_0, BoundVector) and isinstance(object_1, BoundVector):
        intersection = _get_intersection_bound_vector_bound_vector(
            bound_vector_0=object_0,
            bound_vector_1=object_1)
    elif isinstance(object_0, BoundVector) and isinstance(object_1, Plane):
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


def _get_intersection_bound_vector_bound_vector(
        bound_vector_0, bound_vector_1):
    vectors_are_parallel = are_parallel(
        bound_vector_0.free_vector, bound_vector_1.free_vector)

    initial_point_vector = BoundVector(
        initial_point=bound_vector_0.initial_point,
        terminal_point=bound_vector_1.initial_point)

    vectors_in_plane = bool(isclose(
        det(stack(
            [
                bound_vector_0.free_vector,
                bound_vector_1.free_vector,
                initial_point_vector.free_vector
            ],
            axis=-1)),
        0))

    vectors_in_line = (
            bool(isclose(
                norm(
                    cross(
                        initial_point_vector.free_vector,
                        bound_vector_0.free_vector)),
                0,
                **config['numbers_close_kwargs'])) and
            bool(isclose(
                norm(
                    cross(
                        initial_point_vector.free_vector,
                        bound_vector_1.free_vector)),
                0,
                **config['numbers_close_kwargs'])) and
            vectors_are_parallel)

    if vectors_in_line:
        # TODO: work with get_intersection(Point, BoundVector), should it
        # ever exist
        with errstate(invalid='ignore'):
            bv0_bv1_ip_param = nanmean(
                (bound_vector_1.initial_point -
                 bound_vector_0.initial_point) /
                bound_vector_0.free_vector)
            bv0_bv1_tp_param = nanmean(
                (bound_vector_1.terminal_point -
                 bound_vector_0.initial_point) /
                bound_vector_0.free_vector)

        params = sorted([bv0_bv1_ip_param, bv0_bv1_tp_param, 0, 1])

        if (
                (
                    params[1] < 0 and
                    not isclose(
                        params[1],
                        0,
                        **config['numbers_close_kwargs'])) or
                (
                    params[-2] > 1 and
                    not isclose(
                        params[-2],
                        1,
                        **config['numbers_close_kwargs']))):
            intersection = None
        elif are_antiparallel(
                bound_vector_0.free_vector, bound_vector_1.free_vector):
            new_end_point_0_param = params[1]
            new_end_point_1_param = params[-2]
            if isclose(
                    new_end_point_0_param,
                    1,
                    **config['numbers_close_kwargs']):
                # If we are here it means that the tip of bound_vector_0
                # touches the tip of bound_vector_1
                if (
                        bound_vector_0.terminal_point_included and
                        bound_vector_1.terminal_point_included):
                    intersection = bound_vector_0.terminal_point
                else:
                    intersection = None
            elif isclose(
                    new_end_point_1_param,
                    0,
                    **config['numbers_close_kwargs']):
                # If we are here it means that the origin of bound_vector_0
                # touches the origin of bound_vector_0
                if (
                        bound_vector_0.initial_point_included and
                        bound_vector_1.initial_point_included):
                    intersection = bound_vector_0.initial_point
                else:
                    intersection = None
            else:
                # If we are here it means that there is an overlap
                new_end_point_0 = (
                    bound_vector_0.initial_point +
                    new_end_point_0_param*bound_vector_0.free_vector)
                new_end_point_1 = (
                    bound_vector_0.initial_point +
                    new_end_point_1_param*bound_vector_0.free_vector)

                if isclose(
                        new_end_point_0_param,
                        0,
                        **config['numbers_close_kwargs']):
                    if isclose(
                            params[0],
                            0,
                            **config['numbers_close_kwargs']):
                        new_end_point_0_included = (
                            bound_vector_0.initial_point_included and
                            bound_vector_1.terminal_point_included)
                    else:
                        new_end_point_0_included = (
                            bound_vector_0.initial_point_included)
                else:
                    new_end_point_0_included = (
                        bound_vector_1.terminal_point_included)

                if isclose(
                        new_end_point_1_param,
                        1,
                        **config['numbers_close_kwargs']):
                    if isclose(
                            params[-1],
                            1,
                            **config['numbers_close_kwargs']):
                        new_end_point_1_included = (
                            bound_vector_0.terminal_point_included and
                            bound_vector_1.initial_point_included)
                    else:
                        new_end_point_1_included = (
                            bound_vector_0.terminal_point_included)
                else:
                    new_end_point_1_included = (
                        bound_vector_1.initial_point_included)

                intersection = LineSegment(
                    end_point_0=new_end_point_0,
                    end_point_1=new_end_point_1,
                    end_point_0_included=new_end_point_0_included,
                    end_point_1_included=new_end_point_1_included)
        else:
            new_initial_point_param = params[1]
            new_terminal_point_param = params[-2]
            if isclose(
                    new_initial_point_param,
                    1,
                    **config['numbers_close_kwargs']):
                # If we are here it means that the tip of bound_vector_0
                # touches the origin of bound_vector_1
                if (
                        bound_vector_0.terminal_point_included and
                        bound_vector_1.initial_point_included):
                    intersection = bound_vector_0.terminal_point
                else:
                    intersection = None
            elif isclose(
                    new_terminal_point_param,
                    0,
                    **config['numbers_close_kwargs']):
                # If we are here it means that the tip of bound_vector_1
                # touches the origin of bound_vector_0
                if (
                        bound_vector_0.initial_point_included and
                        bound_vector_1.terminal_point_included):
                    intersection = bound_vector_0.initial_point
                else:
                    intersection = None
            else:
                # If we are here it means that there is an overlap
                new_initial_point = (
                    bound_vector_0.initial_point +
                    new_initial_point_param*bound_vector_0.free_vector)
                new_terminal_point = (
                    bound_vector_0.initial_point +
                    new_terminal_point_param*bound_vector_0.free_vector)

                if isclose(
                        new_initial_point_param,
                        0,
                        **config['numbers_close_kwargs']):
                    if isclose(params[0], 0, **config['numbers_close_kwargs']):
                        new_initial_point_included = (
                            bound_vector_0.initial_point_included and
                            bound_vector_1.initial_point_included)
                    else:
                        new_initial_point_included = (
                            bound_vector_0.initial_point_included)
                else:
                    new_initial_point_included = (
                        bound_vector_1.initial_point_included)

                if isclose(
                        new_terminal_point_param,
                        1,
                        **config['numbers_close_kwargs']):
                    if isclose(
                            params[-1],
                            1,
                            **config['numbers_close_kwargs']):
                        new_terminal_point_included = (
                            bound_vector_0.terminal_point_included and
                            bound_vector_1.terminal_point_included)
                    else:
                        new_terminal_point_included = (
                            bound_vector_0.terminal_point_included)
                else:
                    new_terminal_point_included = (
                        bound_vector_1.terminal_point_included)

                intersection = BoundVector(
                    initial_point=new_initial_point,
                    terminal_point=new_terminal_point,
                    initial_point_included=new_initial_point_included,
                    terminal_point_included=new_terminal_point_included)
    elif not vectors_are_parallel and vectors_in_plane:
        denominator = norm(
            cross(bound_vector_0.free_vector, bound_vector_1.free_vector))
        param_0 = (
            norm(
                cross(
                    bound_vector_1.free_vector,
                    initial_point_vector.free_vector)) /
            denominator)
        param_1 = (
            norm(
                cross(
                    bound_vector_0.free_vector,
                    initial_point_vector.free_vector)) /
            denominator)

        if are_antiparallel(
                cross(
                    bound_vector_1.free_vector,
                    bound_vector_1.initial_point -
                    bound_vector_0.initial_point),
                cross(
                    bound_vector_1.free_vector,
                    bound_vector_0.free_vector)):
            param_0 *= -1

        if are_antiparallel(
                cross(
                    bound_vector_0.free_vector,
                    bound_vector_0.initial_point -
                    bound_vector_1.initial_point),
                cross(
                    bound_vector_0.free_vector,
                    bound_vector_1.free_vector)):
            param_1 *= -1

        if bound_vector_0.initial_point_included:
            intersection_ahead_of_bound_vector_0_initial_point = (
                param_0 >= 0 - config['numbers_close_kwargs']['atol'])
        else:
            intersection_ahead_of_bound_vector_0_initial_point = (
                param_0 > 0 - config['numbers_close_kwargs']['atol'])

        if bound_vector_0.terminal_point_included:
            intersection_behind_of_bound_vector_0_terminal_point = (
                param_0 <= 1 + config['numbers_close_kwargs']['atol'])
        else:
            intersection_behind_of_bound_vector_0_terminal_point = (
                param_0 < 1 + config['numbers_close_kwargs']['atol'])

        if bound_vector_1.initial_point_included:
            intersection_ahead_of_bound_vector_1_initial_point = (
                param_1 >= 0 - config['numbers_close_kwargs']['atol'])
        else:
            intersection_ahead_of_bound_vector_1_initial_point = (
                param_1 > 0 - config['numbers_close_kwargs']['atol'])

        if bound_vector_1.terminal_point_included:
            intersection_behind_of_bound_vector_1_terminal_point = (
                param_1 <= 1 + config['numbers_close_kwargs']['atol'])
        else:
            intersection_behind_of_bound_vector_1_terminal_point = (
                param_1 < 1 + config['numbers_close_kwargs']['atol'])

        if (
                intersection_ahead_of_bound_vector_0_initial_point and
                intersection_behind_of_bound_vector_0_terminal_point and
                intersection_ahead_of_bound_vector_1_initial_point and
                intersection_behind_of_bound_vector_1_terminal_point):
            bound_vector_0_intersection = (
                bound_vector_0.initial_point +
                param_0 * bound_vector_0.free_vector)
            bound_vector_1_intersection = (
                bound_vector_1.initial_point +
                param_1 * bound_vector_1.free_vector)

            # Just to be on the safe side, make sure they are actually the
            # same
            assert allclose(
                bound_vector_0_intersection,
                bound_vector_1_intersection,
                **config['numbers_close_kwargs'])

            intersection = bound_vector_0_intersection
        else:
            intersection = None
    else:
        intersection = None
    return intersection


def _get_intersection_bound_vector_plane(bound_vector, plane):
    distance_to_plane = dot(
        plane.point_in_plane - bound_vector.initial_point,
        plane.normal_vector)
    projected_vector_length = dot(
        bound_vector.free_vector,
        plane.normal_vector)

    distance_to_plane_close_to_zero = isclose(
        distance_to_plane,
        0,
        **config['numbers_close_kwargs'])
    projected_vector_length_close_to_zero = isclose(
        projected_vector_length,
        0,
        **config['numbers_close_kwargs'])
    if (
            distance_to_plane_close_to_zero and
            projected_vector_length_close_to_zero):
        return bound_vector

    param = nan_to_num(distance_to_plane / projected_vector_length)

    # TODO: add distinction for included and excluded initial and terminal
    # points
    if 0 <= param <= 1:
        intersection = (
            bound_vector.initial_point +
            param*(bound_vector.terminal_point - bound_vector.initial_point))
    else:
        intersection = None
    return intersection
