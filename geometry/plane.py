# -*- coding: utf-8 -*-
from numpy import asanyarray, isclose


class Plane(object):

    def __init__(self, point_in_plane, normal_vector):
        self.point_in_plane = asanyarray(point_in_plane)
        self.normal_vector = asanyarray(normal_vector)

    def contains(self, point):
        return isclose(
            (asanyarray(point)-self.point_in_plane).dot(self.normal_vector), 0)
