# -*- coding: utf-8 -*-
import pytest
from numpy import array
from numpy.testing import assert_array_equal

from python_geometry.plane import Plane


class TestContains:

    @pytest.mark.parametrize(
        ('plane', 'point', 'contained'),
        [
            (Plane((0, 0, 0), (0, 0, 1)), (0, 0, 0), True),
            (Plane((0, 0, 0), (0, 0, 1)), (1, 0, 0), True),
            (Plane((0, 0, 0), (0, 0, 1)), (0, 1, 0), True),
            (Plane((0, 0, 0), (0, 0, 1)), (0, 0, 1), False),
            (Plane((0, 0, 1), (0, 0, 1)), (0, 0, 0), False),
            (Plane((0, 0, 1), (0, 0, 1)), (0, 0, 1), True),
            (Plane((0, 0, 1), (0, 0, 1)), (0, 0, 2), False),
        ]
    )
    def test_GivenPlaneAndGivenPoint_ReturnIfPlaneContainsPoint(self, plane, point, contained):
        assert plane.contains(point) == contained

    def test_GivenPlaneAndGivenPointArray_ReturnIfPlaneContainsPoints(self):
        plane = Plane((1, 2, 3), (1, 1, 1))
        points = array(
            [
                (1, 2, 3),
                (0, 0, 0),
                (2, 2, 2)
            ])
        contained = array(
            [
                True,
                False,
                True
            ])
        assert_array_equal(plane.contains(points), contained)
