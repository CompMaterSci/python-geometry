# -*- coding: utf-8 -*-
from numpy import allclose, asanyarray

from .config import config


__all__ = ['LineSegment']


class LineSegment(object):

    def __init__(
            self,
            end_point_0,
            end_point_1,
            end_point_0_included=True,
            end_point_1_included=True):
        self._end_point_0 = None
        self._end_point_1 = None
        self._end_point_0_included = None
        self._end_point_1_included = None

        self.end_point_0 = end_point_0
        self.end_point_1 = end_point_1
        self.end_point_0_included = end_point_0_included
        self.end_point_1_included = end_point_1_included

    @property
    def end_point_0(self):
        return self._end_point_0

    @end_point_0.setter
    def end_point_0(self, value):
        self._end_point_0 = asanyarray(value)

    @property
    def end_point_1(self):
        return self._end_point_1

    @end_point_1.setter
    def end_point_1(self, value):
        self._end_point_1 = asanyarray(value)

    @property
    def end_point_0_included(self):
        return self._end_point_0_included

    @end_point_0_included.setter
    def end_point_0_included(self, value):
        self._end_point_0_included = bool(value)

    @property
    def end_point_1_included(self):
        return self._end_point_1_included

    @end_point_1_included.setter
    def end_point_1_included(self, value):
        self._end_point_1_included = bool(value)

    def __eq__(self, other):
        self_points = sorted(
            [self.end_point_0.tolist(), self.end_point_1.tolist()])
        other_points = sorted(
            [other.end_point_0.tolist(), other.end_point_1.tolist()])
        return (
            isinstance(other, LineSegment) and
            allclose(
                other_points,
                self_points,
                **config['numbers_close_kwargs']) and
            other.end_point_0_included is self.end_point_0_included and
            other.end_point_1_included is self.end_point_1_included)

    def __repr__(self):
        return (
            '{self.__class__.__name__}('
            'end_point_0={self.end_point_0!r}, '
            'end_point_1={self.end_point_1!r}, '
            'end_point_0_included={self.end_point_0_included!r}, '
            'end_point_1_included={self.end_point_1_included!r}'
            ')').format(self=self)
