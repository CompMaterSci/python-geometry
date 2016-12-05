# -*- coding: utf-8 -*-
import re

from numpy import all, allclose, asanyarray, isclose, copy, roll, where

from .config import config
from .utilities import get_normal_vector


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
