# -*- coding: utf-8 -*-
from numpy.testing import assert_allclose

from python_geometry.utilities import get_normal_vector


class TestGetNormalVector:

    def test_SquareInXYPlanePositiveDirection_ReturnCorrectNormalVector(self):
        points = [
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0)
        ]
        assert_allclose(get_normal_vector(points), (0, 0, 1))

    def test_SquareInXYPlaneNegativeDirection_ReturnCorrectNormalVector(self):
        points = [
            (0, 0, 0),
            (0, 1, 0),
            (1, 1, 0),
            (1, 0, 0)
        ]
        assert_allclose(get_normal_vector(points), (0, 0, -1))

    def test_SquareInXYPlaneMostlyPositiveDirection_ReturnCorrectNormalVector(self):
        points = [
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 2, 0),
            (-1, 2, 0),
            (-1, 1, 0)
        ]
        assert_allclose(get_normal_vector(points), (0, 0, 1))

    def test_StraightLine_ReturnZeroVector(self):
        points = [
            (0, 0, 0),
            (1, 1, 1),
            (2, 2, 2)
        ]
        assert_allclose(get_normal_vector(points), (0, 0, 0))

    def test_StraightLineWithTwoPoints_ReturnZeroVector(self):
        points = [
            (0, 0, 0),
            (1, 1, 1)
        ]
        assert_allclose(get_normal_vector(points), (0, 0, 0))
