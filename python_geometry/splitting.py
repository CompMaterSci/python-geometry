# -*- coding: utf-8 -*-
from itertools import chain
from copy import copy

from numpy import allclose, int32, mean, sign

from .simple_polygon import SimplePolygon
from .utilities import get_intersection
from .config import config
from .bound_vector import BoundVector
from .line_segment import LineSegment


def split_by_plane(object_to_split, plane):
    if isinstance(object_to_split, SimplePolygon):
        split_objects = _split_polygon_by_plane(
            polygon_to_split=object_to_split,
            plane=plane)
    else:
        raise NotImplementedError(
            'Splitting a "{}" by a plane is not yet implemented.')
    return split_objects


def _split_polygon_by_plane(polygon_to_split, plane):
    vertices_list = [[]]
    for bound_vector in polygon_to_split.bound_vectors:
        intersection = get_intersection(bound_vector, plane)
        if (
                isinstance(intersection, BoundVector) or
                isinstance(intersection, LineSegment)):
            continue

        vertices_list[-1].append(bound_vector.initial_point)

        # If there is no intersection we do not have to take any further
        # actions
        if intersection is None:
            continue

        # If the initial point coincides with the intersection we do not have
        # to do anything
        if allclose(
                bound_vector.initial_point,
                intersection,
                **config['numbers_close_kwargs']):
            continue

        vertices_list[-1].append(intersection)

        # If the terminal point coincides with the intersection just provide a
        # new empty list; the next initial point will be added anyway
        if allclose(
                bound_vector.terminal_point,
                intersection,
                **config['numbers_close_kwargs']):
            vertices_list.append([])
            continue

        vertices_list.append([intersection])

    # The last and the first part may belong to one and the same polygon
    if (len(vertices_list) > 1 and
            allclose(
                bound_vector.terminal_point,
                polygon_to_split.vertices[0],
                **config['numbers_close_kwargs'])):
        last_vertices = vertices_list.pop()
        vertices_list[0] = last_vertices + vertices_list[0]

    i_that_needs_merging = None
    j_that_are_merged = []
    for i, vertices_i in enumerate(vertices_list):
        side_i = sign(mean(plane.distance(vertices_i))).astype(int32)
        other_vertices = chain(
            enumerate(vertices_list[i+1:], start=i+1),
            enumerate(vertices_list[:i], start=0))
        for j, vertices_j in other_vertices:
            side_j = sign(mean(plane.distance(vertices_j))).astype(int32)
            if side_j == side_i:
                bound_vector_i = BoundVector(
                    initial_point=vertices_i[-1],
                    terminal_point=vertices_i[0])
                bound_vector_j = BoundVector(
                    initial_point=vertices_j[-1],
                    terminal_point=vertices_j[0])
                intersection = get_intersection(bound_vector_i, bound_vector_j)
                if (
                        isinstance(intersection, LineSegment) or
                        isinstance(intersection, BoundVector)):
                    i_that_needs_merging = i
                    j_that_are_merged.append(j)
        if i_that_needs_merging is not None:
            break

    new_vertices_list = []
    for i, vertices in enumerate(vertices_list):
        if i in j_that_are_merged:
            continue
        if i == i_that_needs_merging:
            merged_vertices = copy(vertices)
            for j in j_that_are_merged:
                merged_vertices.extend(vertices_list[j])
            new_vertices_list.append(merged_vertices)
        else:
            new_vertices_list.append(vertices)

    # Return proper polygons
    polygons = []
    for vertices in new_vertices_list:
        polygons.append(SimplePolygon(
            vertices=vertices))

    return polygons
