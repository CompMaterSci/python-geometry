# -*- coding: utf-8 -*-
import re
from itertools import tee

from numpy import all, allclose, asanyarray, isclose, copy, roll, where

from .config import config
from .utilities import get_normal_vector
from .bound_vector import BoundVector


def _pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class SimplePolygon(object):

    def __init__(self, vertices):
        self._vertices = None

        self.vertices = vertices

    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, value):
        self._vertices = asanyarray(value)

    @property
    def normal_vector(self):
        vertex_list = self.vertices.tolist()
        return get_normal_vector(vertex_list + [vertex_list[0]])

    @property
    def bound_vectors(self):
        for initial_point, terminal_point in _pairwise(self.vertices):
            yield BoundVector(
                initial_point=initial_point,
                terminal_point=terminal_point)
        yield BoundVector(
            initial_point=self.vertices[-1],
            terminal_point=self.vertices[0])

    def __eq__(self, other):
        try:
            other_vertices = copy(other.vertices)
        except AttributeError:
            return False

        # Only the order of the vertices is important, not start or end
        mask = where(all(
            isclose(
                other_vertices,
                self.vertices[0],
                **config['numbers_close_kwargs']),
            axis=1))[0]
        try:
            i = mask[0]
        except IndexError:
            return False
        other_vertices = roll(other_vertices, -i, axis=0)

        try:
            return allclose(
                other_vertices,
                self.vertices,
                **config['numbers_close_kwargs'])
        except ValueError:
            return False

    def __repr__(self):
        return re.sub('\s+', ' ', (
            '{self.__class__.__name__}('
            'vertices={self.vertices!r}'
            ')').format(self=self))
