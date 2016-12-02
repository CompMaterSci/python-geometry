# -*- coding: utf-8 -*-
import pytest
from numpy.testing import assert_allclose
from numpy import array

from python_geometry.utilities import get_intersection, get_normal_vector
from python_geometry.plane import Plane
from python_geometry.bound_vector import BoundVector


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

    def test_BoundVectorBoundVectorIntersection_RaiseNotImplementedError(self):
        bound_vector_0 = BoundVector(
            initial_point=array([-1, 0, 0]),
            terminal_point=array([1, 0, 0]))
        bound_vector_1 = BoundVector(
            initial_point=array([-1, 0, 0]),
            terminal_point=array([1, 0, 0]))
        with pytest.raises(NotImplementedError):
            get_intersection(bound_vector_0, bound_vector_1)
