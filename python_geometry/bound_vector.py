# -*- coding: utf-8 -*-
from numpy import allclose, asanyarray

from .config import config


__all__ = ['BoundVector']


class BoundVector(object):

    def __init__(
            self,
            initial_point,
            terminal_point,
            initial_point_included=True,
            terminal_point_included=True):
        self._initial_point = None
        self._terminal_point = None
        self._initial_point_included = None
        self._terminal_point_included = None

        self.initial_point = initial_point
        self.terminal_point = terminal_point
        self.initial_point_included = initial_point_included
        self.terminal_point_included = terminal_point_included

    @property
    def initial_point(self):
        return self._initial_point

    @initial_point.setter
    def initial_point(self, value):
        self._initial_point = asanyarray(value)

    @property
    def terminal_point(self):
        return self._terminal_point

    @terminal_point.setter
    def terminal_point(self, value):
        self._terminal_point = asanyarray(value)

    @property
    def initial_point_included(self):
        return self._initial_point_included

    @initial_point_included.setter
    def initial_point_included(self, value):
        self._initial_point_included = bool(value)

    @property
    def terminal_point_included(self):
        return self._terminal_point_included

    @terminal_point_included.setter
    def terminal_point_included(self, value):
        self._terminal_point_included = bool(value)

    def __eq__(self, other):
        return (
            isinstance(other, BoundVector) and
            allclose(
                other.initial_point,
                self.initial_point,
                **config['numbers_close_kwargs']) and
            allclose(
                other.terminal_point,
                self.terminal_point,
                **config['numbers_close_kwargs']) and
            other.initial_point_included is self.initial_point_included and
            other.terminal_point_included is self.terminal_point_included)

    def __repr__(self):
        return (
            '{self.__class__.__name__}('
            'initial_point={self.initial_point!r}, '
            'terminal_point={self.terminal_point!r}, '
            'initial_point_included={self.initial_point_included!r}, '
            'terminal_point_included={self.terminal_point_included!r}'
            ')').format(self=self)
