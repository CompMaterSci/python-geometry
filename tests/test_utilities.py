# -*- coding: utf-8 -*-
import pytest
from numpy.testing import assert_allclose
from numpy import array

from python_geometry.utilities import get_intersection, get_normal_vector
from python_geometry.plane import Plane
from python_geometry.bound_vector import BoundVector
from python_geometry.line_segment import LineSegment


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


class TestGetIntersection(object):

    @pytest.mark.parametrize(
        (
            'bound_vector_0_initial_point', 'bound_vector_0_terminal_point',
            'bound_vector_1_initial_point', 'bound_vector_1_terminal_point',
            'expected'
        ),
        [
            (
                array([0, 1, 0]), array([2, 1, 0]),
                array([1, 0, 0]), array([1, 2, 0]),
                array([1, 1, 0])
            ),
            (
                array([0, 1, 0]), array([2, 1, 0]),
                array([0, 0, 0]), array([0, 2, 0]),
                array([0, 1, 0])
            ),
            (
                array([0, 1, 0]), array([2, 1, 0]),
                array([2, 0, 0]), array([2, 2, 0]),
                array([2, 1, 0])
            ),
            (
                array([0, 0, 0]), array([2, 0, 0]),
                array([1, 0, 0]), array([1, 2, 0]),
                array([1, 0, 0])
            ),
            (
                array([0, 2, 0]), array([2, 2, 0]),
                array([1, 0, 0]), array([1, 2, 0]),
                array([1, 2, 0])
            ),
            (
                array([0, 0, 0]), array([2, 0, 0]),
                array([0, 0, 0]), array([0, 2, 0]),
                array([0, 0, 0])
            ),
            (
                array([0, 0, 0]), array([2, 0, 0]),
                array([2, 0, 0]), array([2, 2, 0]),
                array([2, 0, 0])
            ),
            (
                array([0, 2, 0]), array([2, 2, 0]),
                array([0, 0, 0]), array([0, 2, 0]),
                array([0, 2, 0])
            ),
            (
                array([0, 2, 0]), array([2, 2, 0]),
                array([2, 0, 0]), array([2, 2, 0]),
                array([2, 2, 0])
            )
        ])
    def test_BoundVectorIntersectsNonParallelBoundVectorOnSamePlane_ReturnPoint(
            self, bound_vector_0_initial_point, bound_vector_0_terminal_point,
            bound_vector_1_initial_point, bound_vector_1_terminal_point,
            expected):
        bound_vector_0 = BoundVector(
            initial_point=bound_vector_0_initial_point,
            terminal_point=bound_vector_0_terminal_point,
            initial_point_included=True,
            terminal_point_included=True)
        bound_vector_1 = BoundVector(
            initial_point=bound_vector_1_initial_point,
            terminal_point=bound_vector_1_terminal_point,
            initial_point_included=True,
            terminal_point_included=True)
        actual = get_intersection(bound_vector_0, bound_vector_1)
        assert_allclose(actual, expected)

    @pytest.mark.parametrize(
        (
            'bound_vector_0_initial_point', 'bound_vector_0_terminal_point',
            'bound_vector_1_initial_point', 'bound_vector_1_terminal_point'
        ),
        [
            (
                array([1, 0, 0]), array([2, 0, 0]),
                array([0, 1, 0]), array([0, 2, 0])
            )
        ])
    def test_BoundVectorDoesNotIntersectNonParallelBoundVectorInSamePlane_ReturnNone(
            self, bound_vector_0_initial_point, bound_vector_0_terminal_point,
            bound_vector_1_initial_point, bound_vector_1_terminal_point):
        bound_vector_0 = BoundVector(
            initial_point=bound_vector_0_initial_point,
            terminal_point=bound_vector_0_terminal_point,
            initial_point_included=True,
            terminal_point_included=True)
        bound_vector_1 = BoundVector(
            initial_point=bound_vector_1_initial_point,
            terminal_point=bound_vector_1_terminal_point,
            initial_point_included=True,
            terminal_point_included=True)
        actual = get_intersection(bound_vector_0, bound_vector_1)
        assert actual is None

    @pytest.mark.parametrize(
        (
            'bound_vector_0_initial_point', 'bound_vector_0_terminal_point',
            'bound_vector_1_initial_point', 'bound_vector_1_terminal_point'
        ),
        [
            (
                array([0, 0, 0]), array([1, 0, 0]),
                array([2, 0, 0]), array([3, 0, 0])
            )
        ])
    def test_BoundVectorDoesNotIntersectParallelBoundVectorOnSameLine_ReturnNone(
            self, bound_vector_0_initial_point, bound_vector_0_terminal_point,
            bound_vector_1_initial_point, bound_vector_1_terminal_point):
        bound_vector_0 = BoundVector(
            initial_point=bound_vector_0_initial_point,
            terminal_point=bound_vector_0_terminal_point,
            initial_point_included=True,
            terminal_point_included=True)
        bound_vector_1 = BoundVector(
            initial_point=bound_vector_1_initial_point,
            terminal_point=bound_vector_1_terminal_point,
            initial_point_included=True,
            terminal_point_included=True)
        actual = get_intersection(bound_vector_0, bound_vector_1)
        assert actual is None

    @pytest.mark.parametrize(
        (
            'bound_vector_0_initial_point', 'bound_vector_0_terminal_point',
            'bound_vector_1_initial_point', 'bound_vector_1_terminal_point',
            'expected'
        ),
        [
            (
                array([0, 0, 0]), array([1, 0, 0]),
                array([1, 0, 0]), array([2, 0, 0]),
                array([1, 0, 0])
            ),
            (
                array([0, 0, 0]), array([1, 0, 0]),
                array([2, 0, 0]), array([1, 0, 0]),
                array([1, 0, 0])
            ),
            (
                array([1, 0, 0]), array([0, 0, 0]),
                array([1, 0, 0]), array([2, 0, 0]),
                array([1, 0, 0])
            ),
            (
                array([1, 0, 0]), array([0, 0, 0]),
                array([2, 0, 0]), array([1, 0, 0]),
                array([1, 0, 0])
            )
        ])
    def test_BoundVectorIntersectsParallelBoundVectorOnSameLineInOnePoint_ReturnPoint(
            self, bound_vector_0_initial_point, bound_vector_0_terminal_point,
            bound_vector_1_initial_point, bound_vector_1_terminal_point,
            expected):
        bound_vector_0 = BoundVector(
            initial_point=bound_vector_0_initial_point,
            terminal_point=bound_vector_0_terminal_point,
            initial_point_included=True,
            terminal_point_included=True)
        bound_vector_1 = BoundVector(
            initial_point=bound_vector_1_initial_point,
            terminal_point=bound_vector_1_terminal_point,
            initial_point_included=True,
            terminal_point_included=True)
        actual = get_intersection(bound_vector_0, bound_vector_1)
        assert_allclose(actual, expected)

    @pytest.mark.parametrize(
        (
            'bound_vector_0_initial_point', 'bound_vector_0_terminal_point', 'bound_vector_0_initial_point_included', 'bound_vector_0_terminal_point_included',
            'bound_vector_1_initial_point', 'bound_vector_1_terminal_point', 'bound_vector_1_initial_point_included', 'bound_vector_1_terminal_point_included',
            'expected_initial_point', 'expected_terminal_point', 'expected_initial_point_included', 'expected_terminal_point_included'
        ),
        [
            (
                array([0, 0, 0]), array([2, 0, 0]), True, True,
                array([1, 0, 0]), array([3, 0, 0]), True, True,
                array([1, 0, 0]), array([2, 0, 0]), True, True
            ),
            (
                array([0, 0, 0]), array([2, 0, 0]), False, True,
                array([1, 0, 0]), array([3, 0, 0]), True, True,
                array([1, 0, 0]), array([2, 0, 0]), True, True
            ),
            (
                array([0, 0, 0]), array([2, 0, 0]), True, False,
                array([1, 0, 0]), array([3, 0, 0]), True, True,
                array([1, 0, 0]), array([2, 0, 0]), True, False
            ),
            (
                array([0, 0, 0]), array([2, 0, 0]), True, True,
                array([1, 0, 0]), array([3, 0, 0]), False, True,
                array([1, 0, 0]), array([2, 0, 0]), False, True
            ),
            (
                array([0, 0, 0]), array([2, 0, 0]), True, True,
                array([1, 0, 0]), array([3, 0, 0]), True, False,
                array([1, 0, 0]), array([2, 0, 0]), True, True
            ),
            (
                array([0, 0, 0]), array([2, 0, 0]), False, False,
                array([1, 0, 0]), array([3, 0, 0]), True, True,
                array([1, 0, 0]), array([2, 0, 0]), True, False
            ),
            (
                array([0, 0, 0]), array([2, 0, 0]), False, True,
                array([1, 0, 0]), array([3, 0, 0]), False, True,
                array([1, 0, 0]), array([2, 0, 0]), False, True
            ),
            (
                array([0, 0, 0]), array([2, 0, 0]), False, True,
                array([1, 0, 0]), array([3, 0, 0]), True, False,
                array([1, 0, 0]), array([2, 0, 0]), True, True
            ),
            (
                array([0, 0, 0]), array([2, 0, 0]), True, False,
                array([1, 0, 0]), array([3, 0, 0]), False, True,
                array([1, 0, 0]), array([2, 0, 0]), False, False
            ),
            (
                array([0, 0, 0]), array([2, 0, 0]), True, False,
                array([1, 0, 0]), array([3, 0, 0]), True, False,
                array([1, 0, 0]), array([2, 0, 0]), True, False
            ),
            (
                array([0, 0, 0]), array([2, 0, 0]), True, True,
                array([1, 0, 0]), array([3, 0, 0]), False, False,
                array([1, 0, 0]), array([2, 0, 0]), False, True
            ),
            (
                array([0, 0, 0]), array([2, 0, 0]), False, False,
                array([1, 0, 0]), array([3, 0, 0]), False, True,
                array([1, 0, 0]), array([2, 0, 0]), False, False
            ),
            (
                array([0, 0, 0]), array([2, 0, 0]), False, False,
                array([1, 0, 0]), array([3, 0, 0]), True, False,
                array([1, 0, 0]), array([2, 0, 0]), True, False
            ),
            (
                array([0, 0, 0]), array([2, 0, 0]), False, True,
                array([1, 0, 0]), array([3, 0, 0]), False, False,
                array([1, 0, 0]), array([2, 0, 0]), False, True
            ),
            (
                array([0, 0, 0]), array([2, 0, 0]), True, False,
                array([1, 0, 0]), array([3, 0, 0]), False, False,
                array([1, 0, 0]), array([2, 0, 0]), False, False
            ),
            # Sharing a common initial point
            # all points included
            (
                array([0, 0, 0]), array([1, 0, 0]), True, True,
                array([0, 0, 0]), array([2, 0, 0]), True, True,
                array([0, 0, 0]), array([1, 0, 0]), True, True,
            ),
            # one point not included
            (
                array([0, 0, 0]), array([1, 0, 0]), False, True,
                array([0, 0, 0]), array([2, 0, 0]), True, True,
                array([0, 0, 0]), array([1, 0, 0]), False, True,
            ),
            (
                array([0, 0, 0]), array([1, 0, 0]), True, False,
                array([0, 0, 0]), array([2, 0, 0]), True, True,
                array([0, 0, 0]), array([1, 0, 0]), True, False,
            ),
            (
                array([0, 0, 0]), array([1, 0, 0]), True, True,
                array([0, 0, 0]), array([2, 0, 0]), False, True,
                array([0, 0, 0]), array([1, 0, 0]), False, True,
            ),
            (
                array([0, 0, 0]), array([1, 0, 0]), True, True,
                array([0, 0, 0]), array([2, 0, 0]), True, False,
                array([0, 0, 0]), array([1, 0, 0]), True, True,
            ),
            # two points not included
            (
                array([0, 0, 0]), array([1, 0, 0]), False, False,
                array([0, 0, 0]), array([2, 0, 0]), True, True,
                array([0, 0, 0]), array([1, 0, 0]), False, False,
            ),
            (
                array([0, 0, 0]), array([1, 0, 0]), False, True,
                array([0, 0, 0]), array([2, 0, 0]), False, True,
                array([0, 0, 0]), array([1, 0, 0]), False, True,
            ),
            (
                array([0, 0, 0]), array([1, 0, 0]), False, True,
                array([0, 0, 0]), array([2, 0, 0]), True, False,
                array([0, 0, 0]), array([1, 0, 0]), False, True,
            ),
            (
                array([0, 0, 0]), array([1, 0, 0]), True, False,
                array([0, 0, 0]), array([2, 0, 0]), False, True,
                array([0, 0, 0]), array([1, 0, 0]), False, False,
            ),
            (
                array([0, 0, 0]), array([1, 0, 0]), True, False,
                array([0, 0, 0]), array([2, 0, 0]), True, False,
                array([0, 0, 0]), array([1, 0, 0]), True, False,
            ),
            (
                array([0, 0, 0]), array([1, 0, 0]), True, True,
                array([0, 0, 0]), array([2, 0, 0]), False, False,
                array([0, 0, 0]), array([1, 0, 0]), False, True,
            ),
            # three points not included
            (
                array([0, 0, 0]), array([1, 0, 0]), True, False,
                array([0, 0, 0]), array([2, 0, 0]), False, False,
                array([0, 0, 0]), array([1, 0, 0]), False, False,
            ),
            (
                array([0, 0, 0]), array([1, 0, 0]), False, True,
                array([0, 0, 0]), array([2, 0, 0]), False, False,
                array([0, 0, 0]), array([1, 0, 0]), False, True,
            ),
            (
                array([0, 0, 0]), array([1, 0, 0]), False, False,
                array([0, 0, 0]), array([2, 0, 0]), True, False,
                array([0, 0, 0]), array([1, 0, 0]), False, False,
            ),
            (
                array([0, 0, 0]), array([1, 0, 0]), False, False,
                array([0, 0, 0]), array([2, 0, 0]), False, True,
                array([0, 0, 0]), array([1, 0, 0]), False, False,
            ),
            # no point included
            (
                array([0, 0, 0]), array([1, 0, 0]), False, False,
                array([0, 0, 0]), array([2, 0, 0]), False, False,
                array([0, 0, 0]), array([1, 0, 0]), False, False,
            ),
            # Sharing a common terminal point
            # all points included
            (
                array([1, 0, 0]), array([2, 0, 0]), True, True,
                array([0, 0, 0]), array([2, 0, 0]), True, True,
                array([1, 0, 0]), array([2, 0, 0]), True, True,
            ),
            # one point not included
            (
                array([1, 0, 0]), array([2, 0, 0]), False, True,
                array([0, 0, 0]), array([2, 0, 0]), True, True,
                array([1, 0, 0]), array([2, 0, 0]), False, True,
            ),
            (
                array([1, 0, 0]), array([2, 0, 0]), True, False,
                array([0, 0, 0]), array([2, 0, 0]), True, True,
                array([1, 0, 0]), array([2, 0, 0]), True, False,
            ),
            (
                array([1, 0, 0]), array([2, 0, 0]), True, True,
                array([0, 0, 0]), array([2, 0, 0]), False, True,
                array([1, 0, 0]), array([2, 0, 0]), True, True,
            ),
            (
                array([1, 0, 0]), array([2, 0, 0]), True, True,
                array([0, 0, 0]), array([2, 0, 0]), True, False,
                array([1, 0, 0]), array([2, 0, 0]), True, False,
            ),
            # two points not included
            (
                array([1, 0, 0]), array([2, 0, 0]), False, False,
                array([0, 0, 0]), array([2, 0, 0]), True, True,
                array([1, 0, 0]), array([2, 0, 0]), False, False,
            ),
            (
                array([1, 0, 0]), array([2, 0, 0]), False, True,
                array([0, 0, 0]), array([2, 0, 0]), False, True,
                array([1, 0, 0]), array([2, 0, 0]), False, True,
            ),
            (
                array([1, 0, 0]), array([2, 0, 0]), False, True,
                array([0, 0, 0]), array([2, 0, 0]), True, False,
                array([1, 0, 0]), array([2, 0, 0]), False, False,
            ),
            (
                array([1, 0, 0]), array([2, 0, 0]), True, False,
                array([0, 0, 0]), array([2, 0, 0]), False, True,
                array([1, 0, 0]), array([2, 0, 0]), True, False,
            ),
            (
                array([1, 0, 0]), array([2, 0, 0]), True, False,
                array([0, 0, 0]), array([2, 0, 0]), True, False,
                array([1, 0, 0]), array([2, 0, 0]), True, False,
            ),
            (
                array([1, 0, 0]), array([2, 0, 0]), True, True,
                array([0, 0, 0]), array([2, 0, 0]), False, False,
                array([1, 0, 0]), array([2, 0, 0]), True, False,
            ),
            # three points not included
            (
                array([1, 0, 0]), array([2, 0, 0]), True, False,
                array([0, 0, 0]), array([2, 0, 0]), False, False,
                array([1, 0, 0]), array([2, 0, 0]), True, False,
            ),
            (
                array([1, 0, 0]), array([2, 0, 0]), False, True,
                array([0, 0, 0]), array([2, 0, 0]), False, False,
                array([1, 0, 0]), array([2, 0, 0]), False, False,
            ),
            (
                array([1, 0, 0]), array([2, 0, 0]), False, False,
                array([0, 0, 0]), array([2, 0, 0]), True, False,
                array([1, 0, 0]), array([2, 0, 0]), False, False,
            ),
            (
                array([1, 0, 0]), array([2, 0, 0]), False, False,
                array([0, 0, 0]), array([2, 0, 0]), False, True,
                array([1, 0, 0]), array([2, 0, 0]), False, False,
            ),
            # no point included
            (
                array([1, 0, 0]), array([2, 0, 0]), False, False,
                array([0, 0, 0]), array([2, 0, 0]), False, False,
                array([1, 0, 0]), array([2, 0, 0]), False, False,
            )
        ])
    def test_BoundVectorIntersectsParallelBoundVectorOnSameLineInMoreThanOnePoint_ReturnBoundVector(
            self,
            bound_vector_0_initial_point, bound_vector_0_terminal_point, bound_vector_0_initial_point_included, bound_vector_0_terminal_point_included,
            bound_vector_1_initial_point, bound_vector_1_terminal_point, bound_vector_1_initial_point_included, bound_vector_1_terminal_point_included,
            expected_initial_point, expected_terminal_point, expected_initial_point_included, expected_terminal_point_included):
        bound_vector_0 = BoundVector(
            initial_point=bound_vector_0_initial_point,
            terminal_point=bound_vector_0_terminal_point,
            initial_point_included=bound_vector_0_initial_point_included,
            terminal_point_included=bound_vector_0_terminal_point_included)
        bound_vector_1 = BoundVector(
            initial_point=bound_vector_1_initial_point,
            terminal_point=bound_vector_1_terminal_point,
            initial_point_included=bound_vector_1_initial_point_included,
            terminal_point_included=bound_vector_1_terminal_point_included)
        actual = get_intersection(bound_vector_0, bound_vector_1)
        expected = BoundVector(
            initial_point=expected_initial_point,
            terminal_point=expected_terminal_point,
            initial_point_included=expected_initial_point_included,
            terminal_point_included=expected_terminal_point_included)
        assert actual == expected

    @pytest.mark.parametrize(
        (
            'bound_vector_0_initial_point', 'bound_vector_0_terminal_point', 'bound_vector_0_initial_point_included', 'bound_vector_0_terminal_point_included',
            'bound_vector_1_initial_point', 'bound_vector_1_terminal_point', 'bound_vector_1_initial_point_included', 'bound_vector_1_terminal_point_included',
            'expected_end_point_0', 'expected_end_point_1', 'expected_end_point_0_included', 'expected_end_point_1_included'
        ),
        [
            (
                array([0, 0, 0]), array([2, 0, 0]), True, True,
                array([3, 0, 0]), array([1, 0, 0]), True, True,
                array([1, 0, 0]), array([2, 0, 0]), True, True
            ),
            (
                array([0, 0, 0]), array([2, 0, 0]), False, True,
                array([3, 0, 0]), array([1, 0, 0]), True, True,
                array([1, 0, 0]), array([2, 0, 0]), True, True
            ),
            (
                array([0, 0, 0]), array([2, 0, 0]), True, False,
                array([3, 0, 0]), array([1, 0, 0]), True, True,
                array([1, 0, 0]), array([2, 0, 0]), True, False
            ),
            (
                array([0, 0, 0]), array([2, 0, 0]), True, True,
                array([3, 0, 0]), array([1, 0, 0]), False, True,
                array([1, 0, 0]), array([2, 0, 0]), True, True
            ),
            (
                array([0, 0, 0]), array([2, 0, 0]), True, True,
                array([3, 0, 0]), array([1, 0, 0]), True, False,
                array([1, 0, 0]), array([2, 0, 0]), False, True
            ),
            (
                array([0, 0, 0]), array([2, 0, 0]), False, False,
                array([3, 0, 0]), array([1, 0, 0]), True, True,
                array([1, 0, 0]), array([2, 0, 0]), True, False
            ),
            (
                array([0, 0, 0]), array([2, 0, 0]), False, True,
                array([3, 0, 0]), array([1, 0, 0]), False, True,
                array([1, 0, 0]), array([2, 0, 0]), True, True
            ),
            (
                array([0, 0, 0]), array([2, 0, 0]), False, True,
                array([3, 0, 0]), array([1, 0, 0]), True, False,
                array([1, 0, 0]), array([2, 0, 0]), False, True
            ),
            (
                array([0, 0, 0]), array([2, 0, 0]), True, False,
                array([3, 0, 0]), array([1, 0, 0]), False, True,
                array([1, 0, 0]), array([2, 0, 0]), True, False
            ),
            (
                array([0, 0, 0]), array([2, 0, 0]), True, False,
                array([3, 0, 0]), array([1, 0, 0]), True, False,
                array([1, 0, 0]), array([2, 0, 0]), False, False
            ),
            (
                array([0, 0, 0]), array([2, 0, 0]), True, True,
                array([3, 0, 0]), array([1, 0, 0]), False, False,
                array([1, 0, 0]), array([2, 0, 0]), False, True
            ),
            (
                array([0, 0, 0]), array([2, 0, 0]), False, False,
                array([3, 0, 0]), array([1, 0, 0]), False, True,
                array([1, 0, 0]), array([2, 0, 0]), True, False
            ),
            (
                array([0, 0, 0]), array([2, 0, 0]), False, False,
                array([3, 0, 0]), array([1, 0, 0]), True, False,
                array([1, 0, 0]), array([2, 0, 0]), False, False
            ),
            (
                array([0, 0, 0]), array([2, 0, 0]), False, True,
                array([3, 0, 0]), array([1, 0, 0]), False, False,
                array([1, 0, 0]), array([2, 0, 0]), False, True
            ),
            (
                array([0, 0, 0]), array([2, 0, 0]), True, False,
                array([3, 0, 0]), array([1, 0, 0]), False, False,
                array([1, 0, 0]), array([2, 0, 0]), False, False
            ),
            # Sharing a common initial point
            # all points included
            (
                array([0, 0, 0]), array([1, 0, 0]), True, True,
                array([2, 0, 0]), array([0, 0, 0]), True, True,
                array([0, 0, 0]), array([1, 0, 0]), True, True,
            ),
            # one point not included
            (
                array([0, 0, 0]), array([1, 0, 0]), False, True,
                array([2, 0, 0]), array([0, 0, 0]), True, True,
                array([0, 0, 0]), array([1, 0, 0]), False, True,
            ),
            (
                array([0, 0, 0]), array([1, 0, 0]), True, False,
                array([2, 0, 0]), array([0, 0, 0]), True, True,
                array([0, 0, 0]), array([1, 0, 0]), True, False,
            ),
            (
                array([0, 0, 0]), array([1, 0, 0]), True, True,
                array([2, 0, 0]), array([0, 0, 0]), False, True,
                array([0, 0, 0]), array([1, 0, 0]), True, True,
            ),
            (
                array([0, 0, 0]), array([1, 0, 0]), True, True,
                array([2, 0, 0]), array([0, 0, 0]), True, False,
                array([0, 0, 0]), array([1, 0, 0]), False, True,
            ),
            # two points not included
            (
                array([0, 0, 0]), array([1, 0, 0]), False, False,
                array([2, 0, 0]), array([0, 0, 0]), True, True,
                array([0, 0, 0]), array([1, 0, 0]), False, False,
            ),
            (
                array([0, 0, 0]), array([1, 0, 0]), False, True,
                array([2, 0, 0]), array([0, 0, 0]), False, True,
                array([0, 0, 0]), array([1, 0, 0]), False, True,
            ),
            (
                array([0, 0, 0]), array([1, 0, 0]), False, True,
                array([2, 0, 0]), array([0, 0, 0]), True, False,
                array([0, 0, 0]), array([1, 0, 0]), False, True,
            ),
            (
                array([0, 0, 0]), array([1, 0, 0]), True, False,
                array([2, 0, 0]), array([0, 0, 0]), False, True,
                array([0, 0, 0]), array([1, 0, 0]), True, False,
            ),
            (
                array([0, 0, 0]), array([1, 0, 0]), True, False,
                array([2, 0, 0]), array([0, 0, 0]), True, False,
                array([0, 0, 0]), array([1, 0, 0]), False, False,
            ),
            (
                array([0, 0, 0]), array([1, 0, 0]), True, True,
                array([2, 0, 0]), array([0, 0, 0]), False, False,
                array([0, 0, 0]), array([1, 0, 0]), False, True,
            ),
            # three points not included
            (
                array([0, 0, 0]), array([1, 0, 0]), True, False,
                array([2, 0, 0]), array([0, 0, 0]), False, False,
                array([0, 0, 0]), array([1, 0, 0]), False, False,
            ),
            (
                array([0, 0, 0]), array([1, 0, 0]), False, True,
                array([2, 0, 0]), array([0, 0, 0]), False, False,
                array([0, 0, 0]), array([1, 0, 0]), False, True,
            ),
            (
                array([0, 0, 0]), array([1, 0, 0]), False, False,
                array([2, 0, 0]), array([0, 0, 0]), True, False,
                array([0, 0, 0]), array([1, 0, 0]), False, False,
            ),
            (
                array([0, 0, 0]), array([1, 0, 0]), False, False,
                array([2, 0, 0]), array([0, 0, 0]), False, True,
                array([0, 0, 0]), array([1, 0, 0]), False, False,
            ),
            # no point included
            (
                array([0, 0, 0]), array([1, 0, 0]), False, False,
                array([2, 0, 0]), array([0, 0, 0]), False, False,
                array([0, 0, 0]), array([1, 0, 0]), False, False,
            ),
            # Sharing a common terminal point
            # all points included
            (
                array([1, 0, 0]), array([2, 0, 0]), True, True,
                array([2, 0, 0]), array([0, 0, 0]), True, True,
                array([1, 0, 0]), array([2, 0, 0]), True, True,
            ),
            # one point not included
            (
                array([1, 0, 0]), array([2, 0, 0]), False, True,
                array([2, 0, 0]), array([0, 0, 0]), True, True,
                array([1, 0, 0]), array([2, 0, 0]), False, True,
            ),
            (
                array([1, 0, 0]), array([2, 0, 0]), True, False,
                array([2, 0, 0]), array([0, 0, 0]), True, True,
                array([1, 0, 0]), array([2, 0, 0]), True, False,
            ),
            (
                array([1, 0, 0]), array([2, 0, 0]), True, True,
                array([2, 0, 0]), array([0, 0, 0]), False, True,
                array([1, 0, 0]), array([2, 0, 0]), True, False,
            ),
            (
                array([1, 0, 0]), array([2, 0, 0]), True, True,
                array([2, 0, 0]), array([0, 0, 0]), True, False,
                array([1, 0, 0]), array([2, 0, 0]), True, True,
            ),
            # two points not included
            (
                array([1, 0, 0]), array([2, 0, 0]), False, False,
                array([2, 0, 0]), array([0, 0, 0]), True, True,
                array([1, 0, 0]), array([2, 0, 0]), False, False,
            ),
            (
                array([1, 0, 0]), array([2, 0, 0]), False, True,
                array([2, 0, 0]), array([0, 0, 0]), False, True,
                array([1, 0, 0]), array([2, 0, 0]), False, False,
            ),
            (
                array([1, 0, 0]), array([2, 0, 0]), False, True,
                array([2, 0, 0]), array([0, 0, 0]), True, False,
                array([1, 0, 0]), array([2, 0, 0]), False, True,
            ),
            (
                array([1, 0, 0]), array([2, 0, 0]), True, False,
                array([2, 0, 0]), array([0, 0, 0]), False, True,
                array([1, 0, 0]), array([2, 0, 0]), True, False,
            ),
            (
                array([1, 0, 0]), array([2, 0, 0]), True, False,
                array([2, 0, 0]), array([0, 0, 0]), True, False,
                array([1, 0, 0]), array([2, 0, 0]), True, False,
            ),
            (
                array([1, 0, 0]), array([2, 0, 0]), True, True,
                array([2, 0, 0]), array([0, 0, 0]), False, False,
                array([1, 0, 0]), array([2, 0, 0]), True, False,
            ),
            # three points not included
            (
                array([1, 0, 0]), array([2, 0, 0]), True, False,
                array([2, 0, 0]), array([0, 0, 0]), False, False,
                array([1, 0, 0]), array([2, 0, 0]), True, False,
            ),
            (
                array([1, 0, 0]), array([2, 0, 0]), False, True,
                array([2, 0, 0]), array([0, 0, 0]), False, False,
                array([1, 0, 0]), array([2, 0, 0]), False, False,
            ),
            (
                array([1, 0, 0]), array([2, 0, 0]), False, False,
                array([2, 0, 0]), array([0, 0, 0]), True, False,
                array([1, 0, 0]), array([2, 0, 0]), False, False,
            ),
            (
                array([1, 0, 0]), array([2, 0, 0]), False, False,
                array([2, 0, 0]), array([0, 0, 0]), False, True,
                array([1, 0, 0]), array([2, 0, 0]), False, False,
            ),
            # no point included
            (
                array([1, 0, 0]), array([2, 0, 0]), False, False,
                array([2, 0, 0]), array([0, 0, 0]), False, False,
                array([1, 0, 0]), array([2, 0, 0]), False, False,
            )
        ])
    def test_BoundVectorIntersectsAntiparallelBoundVectorOnSameLineInMoreThanOnePoint_ReturnLineSegment(
            self,
            bound_vector_0_initial_point, bound_vector_0_terminal_point, bound_vector_0_initial_point_included, bound_vector_0_terminal_point_included,
            bound_vector_1_initial_point, bound_vector_1_terminal_point, bound_vector_1_initial_point_included, bound_vector_1_terminal_point_included,
            expected_end_point_0, expected_end_point_1, expected_end_point_0_included, expected_end_point_1_included):
        bound_vector_0 = BoundVector(
            initial_point=bound_vector_0_initial_point,
            terminal_point=bound_vector_0_terminal_point,
            initial_point_included=bound_vector_0_initial_point_included,
            terminal_point_included=bound_vector_0_terminal_point_included)
        bound_vector_1 = BoundVector(
            initial_point=bound_vector_1_initial_point,
            terminal_point=bound_vector_1_terminal_point,
            initial_point_included=bound_vector_1_initial_point_included,
            terminal_point_included=bound_vector_1_terminal_point_included)
        actual = get_intersection(bound_vector_0, bound_vector_1)
        expected = LineSegment(
            end_point_0=expected_end_point_0,
            end_point_1=expected_end_point_1,
            end_point_0_included=expected_end_point_0_included,
            end_point_1_included=expected_end_point_1_included)
        assert actual == expected

    @pytest.mark.parametrize(
        (
            'bound_vector_0_initial_point', 'bound_vector_0_terminal_point',
            'bound_vector_1_initial_point', 'bound_vector_1_terminal_point'
        ),
        [
            (
                array([0, 0, 0]), array([1, 0, 0]),
                array([0, 1, 0]), array([1, 1, 0])
            ),
            (
                array([0, 0, 0]), array([1, 0, 0]),
                array([0, 0, 1]), array([1, 0, 1])
            )
        ])
    def test_BoundVectorDoesNotIntersectParallelBoundVectorNotOnSameLine_ReturnNone(
            self, bound_vector_0_initial_point, bound_vector_0_terminal_point,
            bound_vector_1_initial_point, bound_vector_1_terminal_point):
        bound_vector_0 = BoundVector(
            initial_point=bound_vector_0_initial_point,
            terminal_point=bound_vector_0_terminal_point,
            initial_point_included=True,
            terminal_point_included=True)
        bound_vector_1 = BoundVector(
            initial_point=bound_vector_1_initial_point,
            terminal_point=bound_vector_1_terminal_point,
            initial_point_included=True,
            terminal_point_included=True)
        actual = get_intersection(bound_vector_0, bound_vector_1)
        assert actual is None

    @pytest.mark.parametrize(
        (
            'bound_vector_0_initial_point', 'bound_vector_0_terminal_point',
            'bound_vector_1_initial_point', 'bound_vector_1_terminal_point'
        ),
        [
            (
                array([0, 1, 0]), array([2, 1, 0]),
                array([1, 0, 1]), array([1, 2, 1])
            )
        ])
    def test_BoundVectorDoesNotIntersectSkewedBoundVector_ReturnNone(
            self, bound_vector_0_initial_point, bound_vector_0_terminal_point,
            bound_vector_1_initial_point, bound_vector_1_terminal_point):
        bound_vector_0 = BoundVector(
            initial_point=bound_vector_0_initial_point,
            terminal_point=bound_vector_0_terminal_point,
            initial_point_included=True,
            terminal_point_included=True)
        bound_vector_1 = BoundVector(
            initial_point=bound_vector_1_initial_point,
            terminal_point=bound_vector_1_terminal_point,
            initial_point_included=True,
            terminal_point_included=True)
        actual = get_intersection(bound_vector_0, bound_vector_1)
        assert actual is None

    def test_PlaneThatIntersectsBoundVectorIntersection_ReturnPoint_0(self):
        plane = Plane(
            point_in_plane=array([0, 0, 0]),
            normal_vector=array([1, 0, 0]))
        bound_vector = BoundVector(
            initial_point=array([-1, 0, 0]),
            terminal_point=array([1, 0, 0]))
        actual = get_intersection(plane, bound_vector)
        expected = array([0, 0, 0])
        assert_allclose(actual, expected)

    def test_PlaneThatIntersectsBoundVectorIntersection_ReturnPoint_1(self):
        plane = Plane(
            point_in_plane=array([1, 1, 1]),
            normal_vector=array([1, 0, 0]))
        bound_vector = BoundVector(
            initial_point=array([-1, 0, 0]),
            terminal_point=array([1, 0, 0]))
        actual = get_intersection(plane, bound_vector)
        expected = array([1, 0, 0])
        assert_allclose(actual, expected)

    def test_PlaneThatDoesNotIntersectBoundVectorIntersection_ReturnNone_0(self):
        plane = Plane(
            point_in_plane=array([0, 0, 0]),
            normal_vector=array([1, 0, 0]))
        bound_vector = BoundVector(
            initial_point=array([1, 0, 0]),
            terminal_point=array([2, 0, 0]))
        assert get_intersection(plane, bound_vector) is None

    def test_PlaneThatDoesNotIntersectBoundVectorIntersection_ReturnNone_1(self):
        plane = Plane(
            point_in_plane=array([0, 0, 0]),
            normal_vector=array([1, 0, 0]))
        bound_vector = BoundVector(
            initial_point=array([-2, 0, 0]),
            terminal_point=array([-1, 0, 0]))
        assert get_intersection(plane, bound_vector) is None

    def test_BoundVectorThatIntersectsPlaneIntersection_ReturnPoint_0(self):
        bound_vector = BoundVector(
            initial_point=array([-1, 0, 0]),
            terminal_point=array([1, 0, 0]))
        plane = Plane(
            point_in_plane=array([0, 0, 0]),
            normal_vector=array([1, 0, 0]))
        actual = get_intersection(bound_vector, plane)
        expected = array([0, 0, 0])
        assert_allclose(actual, expected)

    def test_BoundVectorThatIntersectsPlaneIntersection_ReturnPoint_1(self):
        bound_vector = BoundVector(
            initial_point=array([-1, 0, 0]),
            terminal_point=array([1, 0, 0]))
        plane = Plane(
            point_in_plane=array([1, 1, 1]),
            normal_vector=array([1, 0, 0]))
        actual = get_intersection(bound_vector, plane)
        expected = array([1, 0, 0])
        assert_allclose(actual, expected)

    def test_BoundVectorThatDoesNotIntersectPlaneIntersection_ReturnNone_0(self):
        bound_vector = BoundVector(
            initial_point=array([1, 0, 0]),
            terminal_point=array([2, 0, 0]))
        plane = Plane(
            point_in_plane=array([0, 0, 0]),
            normal_vector=array([1, 0, 0]))
        assert get_intersection(bound_vector, plane) is None


    def test_BoundVectorThatDoesNotIntersectPlaneIntersection_ReturnNone_1(self):
        bound_vector = BoundVector(
            initial_point=array([-2, 0, 0]),
            terminal_point=array([-1, 0, 0]))
        plane = Plane(
            point_in_plane=array([0, 0, 0]),
            normal_vector=array([1, 0, 0]))
        assert get_intersection(bound_vector, plane) is None
