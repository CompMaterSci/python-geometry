# -*- coding: utf-8 -*-
from numpy import array
from numpy.testing import assert_allclose

from python_geometry.bound_vector import BoundVector


class TestEquality(object):

    def test_EqualIntegerBoundVectors_ReturnTrue(self):
        bound_vector_1 = BoundVector(
            initial_point=array([0, 0, 0]),
            terminal_point=array([1, 1, 1]),
            initial_point_included=True,
            terminal_point_included=True)
        bound_vector_2 = BoundVector(
            initial_point=array([0, 0, 0]),
            terminal_point=array([1, 1, 1]),
            initial_point_included=True,
            terminal_point_included=True)
        assert bound_vector_1 == bound_vector_2

    def test_EqualFloatBoundVectors_ReturnTrue(self):
        bound_vector_1 = BoundVector(
            initial_point=array([0.3, 0.3, 0.3]),
            terminal_point=array([1.3, 1.3, 1.3]),
            initial_point_included=True,
            terminal_point_included=True)
        bound_vector_2 = BoundVector(
            initial_point=array([0.3, 0.3, 0.3]),
            terminal_point=array([1.3, 1.3, 1.3]),
            initial_point_included=True,
            terminal_point_included=True)
        assert bound_vector_1 == bound_vector_2

    def test_NonequalIntegerBoundVectors_ReturnTrue_0(self):
        bound_vector_1 = BoundVector(
            initial_point=array([0, 0, 0]),
            terminal_point=array([1, 1, 1]),
            initial_point_included=True,
            terminal_point_included=True)
        bound_vector_2 = BoundVector(
            initial_point=array([0, 0, 0]),
            terminal_point=array([1, 1, 2]),
            initial_point_included=True,
            terminal_point_included=True)
        assert not bound_vector_1 == bound_vector_2

    def test_NonequalIntegerBoundVectors_ReturnTrue_1(self):
        bound_vector_1 = BoundVector(
            initial_point=array([0, 0, 0]),
            terminal_point=array([1, 1, 1]),
            initial_point_included=True,
            terminal_point_included=True)
        bound_vector_2 = BoundVector(
            initial_point=array([0, 0, 0]),
            terminal_point=array([1, 1, 1]),
            initial_point_included=True,
            terminal_point_included=False)
        assert not bound_vector_1 == bound_vector_2

    def test_NonequalIntegerBoundVectors_ReturnTrue_2(self):
        bound_vector_1 = BoundVector(
            initial_point=array([0, 0, 0]),
            terminal_point=array([1, 1, 1]),
            initial_point_included=True,
            terminal_point_included=True)
        bound_vector_2 = BoundVector(
            initial_point=array([0, 0, 0]),
            terminal_point=array([1, 1, 1]),
            initial_point_included=False,
            terminal_point_included=True)
        assert not bound_vector_1 == bound_vector_2

    def test_NonequalFloatBoundVectors_ReturnTrue(self):
        bound_vector_1 = BoundVector(
            initial_point=array([0.3, 0.3, 0.3]),
            terminal_point=array([1.3, 1.3, 1.3]),
            initial_point_included=True,
            terminal_point_included=True)
        bound_vector_2 = BoundVector(
            initial_point=array([0.3, 0.3, 0.3]),
            terminal_point=array([1.3, 1.3, 2.3]),
            initial_point_included=True,
            terminal_point_included=True)
        assert not bound_vector_1 == bound_vector_2


class TestRepr(object):

    def test_CompareAgainstItsRepr_ReturnTrue_0(self):
        bound_vector = BoundVector(
            initial_point=array([0, 0, 0]),
            terminal_point=array([1, 1, 1]))
        assert eval(repr(bound_vector)) == bound_vector

    def test_CompareAgainstItsRepr_ReturnTrue_1(self):
        bound_vector = BoundVector(
            initial_point=array([0, 0, 0]),
            terminal_point=array([1, 1, 1]),
            initial_point_included=True,
            terminal_point_included=True)
        assert eval(repr(bound_vector)) == bound_vector

    def test_CompareAgainstItsRepr_ReturnTrue_2(self):
        bound_vector = BoundVector(
            initial_point=array([0, 0, 0]),
            terminal_point=array([1, 1, 1]),
            initial_point_included=True,
            terminal_point_included=False)
        assert eval(repr(bound_vector)) == bound_vector

    def test_CompareAgainstItsRepr_ReturnTrue_3(self):
        bound_vector = BoundVector(
            initial_point=array([0, 0, 0]),
            terminal_point=array([1, 1, 1]),
            initial_point_included=False,
            terminal_point_included=True)
        assert eval(repr(bound_vector)) == bound_vector

    def test_CompareAgainstItsRepr_ReturnTrue_4(self):
        bound_vector = BoundVector(
            initial_point=array([0, 0, 0]),
            terminal_point=array([1, 1, 1]),
            initial_point_included=False,
            terminal_point_included=False)
        assert eval(repr(bound_vector)) == bound_vector


class TestFreeVector:

    def test_IntegerBoundVector_ReturnCorrectFreeVector_0(self):
        bound_vector = BoundVector(
            initial_point=array([0, 0, 0]),
            terminal_point=array([1, 1, 1]))
        assert_allclose(bound_vector.free_vector, array([1, 1, 1]))

    def test_BoundVector_ReturnCorrectFreeVector_1(self):
        bound_vector = BoundVector(
            initial_point=array([-1, 1.5, 1]),
            terminal_point=array([1, 1, 1]))
        assert_allclose(bound_vector.free_vector, array([2, -0.5, 0]))
