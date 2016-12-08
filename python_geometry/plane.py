# -*- coding: utf-8 -*-
from numpy import asanyarray, dot, isclose

from .config import config


class Plane(object):

    def __init__(self, point_in_plane, normal_vector):
        self.point_in_plane = asanyarray(point_in_plane)
        self.normal_vector = asanyarray(normal_vector)

    def contains(self, point):
        return isclose(
            dot(
                asanyarray(point)-self.point_in_plane,
                self.normal_vector),
            0,
            **config['numbers_close_kwargs'])

    def distance(self, points):
        return dot(points - self.point_in_plane, self.normal_vector)
