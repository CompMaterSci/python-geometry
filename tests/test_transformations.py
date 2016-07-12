# -*- coding: utf-8 -*-
import pytest
from numpy import array, sqrt
from numpy.testing import assert_allclose

from geometry.transformations import (
    _build_row_vector_matrix, Transformation, OrthogonalTransformation)


class TestBuildRowVectorMatrix(object):

    def test_GivenUnitBasis_ReturnIdentityMatrix(self):
        vectors = [
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1)
        ]
        assert_allclose(
            _build_row_vector_matrix(*vectors),
            array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

    def test_GivenMirrorBasis_ReturnCorrectMatrix(self):
        vectors = [
            (0, -1, 0),
            (1, 0, 0),
            (0, 0, 1)
        ]
        assert_allclose(
            _build_row_vector_matrix(*vectors),
            array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]))


class TestTransformation(object):

    def test_repr_GivenUnitBasis_ReturnCorrectRepr(self):
        transformation = Transformation(x=(1, 0, 0), y=(0, 1, 0), z=(0, 0, 1))
        assert (
            repr(transformation) ==
            'Transformation(x=(1, 0, 0), y=(0, 1, 0), z=(0, 0, 1))')

    def test_Pushforward_ComplexExample_ReturnCorrectVectors(self):
        transformation = Transformation(
            x=(-2, 0, 2), y=(-1, 2, -1), z=(-1, -1, -1))
        vectors = (
            (0, 0, 0),
            (1, 0, 0),
            (1.5, -1, 0),
            (1.5, -3, 0),
            (1, -4, 0),
            (0, -4, 0),
            (-0.5, -3, 0),
            (-0.5, -1, 0)
        )
        expected = (
            (+0, +0, +0),
            (-2, +0, +2),
            (-2, -2, +4),
            (+0, -6, +6),
            (+2, -8, +6),
            (+4, -8, +4),
            (+4, -6, +2),
            (+2, -2, +0)
        )
        assert_allclose(
            transformation.pushforward(vectors), expected, atol=1e-8)

    def test_Pullback_ComplexExample_ReturnCorrectVectors(self):
        transformation = Transformation(
            x=(-2, 0, 2), y=(-1, 2, -1), z=(-1, -1, -1))
        vectors = (
            (+0, +0, +0),
            (-2, +0, +2),
            (-2, -2, +4),
            (+0, -6, +6),
            (+2, -8, +6),
            (+4, -8, +4),
            (+4, -6, +2),
            (+2, -2, +0)
        )
        expected = (
            (0, 0, 0),
            (1, 0, 0),
            (1.5, -1, 0),
            (1.5, -3, 0),
            (1, -4, 0),
            (0, -4, 0),
            (-0.5, -3, 0),
            (-0.5, -1, 0)
        )
        assert_allclose(
            transformation.pullback(vectors), expected, atol=1e-8)


class TestOrthogonalTransformation(object):

    def test_Pushforward_GivenUnitBasis_ReturnOriginalVectors(self):
        transformation = OrthogonalTransformation(
            x=(1, 0, 0), y=(0, 1, 0), z=(0, 0, 1))
        vectors = [
            (1, 0, 0),
            (1, 1, 0),
            (1, 1, 1)
        ]
        assert_allclose(transformation.pushforward(vectors), vectors)

    def test_Pushforward_GivenRotatedBasis_ReturnRotatedVectors(self):
        transformation = OrthogonalTransformation(
            x=(0, 1, 0), y=(-1, 0, 0), z=(0, 0, 1))
        vectors = [
            (1, 0, 0),
            (1, 1, 0),
            (1, 1, 1)
        ]
        expected_vectors = [
            (0, 1, 0),
            (-1, 1, 0),
            (-1, 1, 1)
        ]
        assert_allclose(transformation.pushforward(vectors), expected_vectors)
        
    def test_Pushforward_ComplexExample_ReturnCorrectVectors(self):
        transformation = OrthogonalTransformation(
            x=(-2, 0, 2), y=(-1, 2, -1), z=(-1, -1, -1))
        vectors = (
            (+0, +0, +0),
            (+2 * sqrt(2), +0, +0),
            (+3 * sqrt(2), -1 * sqrt(6), +0),
            (+3 * sqrt(2), -3 * sqrt(6), +0),
            (+2 * sqrt(2), -4 * sqrt(6), +0),
            (+0, -4 * sqrt(6), +0),
            (-1 * sqrt(2), -3 * sqrt(6), +0),
            (-1 * sqrt(2), -1 * sqrt(6), +0)
        )
        expected = (
            (+0, +0, +0),
            (-2, +0, +2),
            (-2, -2, +4),
            (+0, -6, +6),
            (+2, -8, +6),
            (+4, -8, +4),
            (+4, -6, +2),
            (+2, -2, +0)
        )
        assert_allclose(
            transformation.pushforward(vectors), expected, atol=1e-8)


    def test_Pushforward_GivenNonUnitBasis_ReturnOriginalVectors(self):
        transformation = OrthogonalTransformation(
            x=(2, 0, 0), y=(0, 3, 0), z=(0, 0, 4))
        vectors = [
            (1, 0, 0),
            (1, 1, 0),
            (1, 1, 1)
        ]
        assert_allclose(transformation.pushforward(vectors), vectors)

    def test_Pushforward_GivenXYUnitBasis_ReturnOriginalVectors(self):
        transformation = OrthogonalTransformation(
            x=(1, 0, 0), y=(0, 1, 0))
        vectors = [
            (1, 0, 0),
            (1, 1, 0),
            (1, 1, 1)
        ]
        assert_allclose(transformation.pushforward(vectors), vectors)

    def test_Pushforward_GivenXYNonUnitBasis_ReturnOriginalVectors(self):
        transformation = OrthogonalTransformation(
            x=(2, 0, 0), y=(0, 3, 0))
        vectors = [
            (1, 0, 0),
            (1, 1, 0),
            (1, 1, 1)
        ]
        assert_allclose(transformation.pushforward(vectors), vectors)

    def test_Pushforward_GivenXZUnitBasis_ReturnOriginalVectors(self):
        transformation = OrthogonalTransformation(
            x=(1, 0, 0), z=(0, 0, 1))
        vectors = [
            (1, 0, 0),
            (1, 1, 0),
            (1, 1, 1)
        ]
        assert_allclose(transformation.pushforward(vectors), vectors)

    def test_Pushforward_GivenXZNonUnitBasis_ReturnOriginalVectors(self):
        transformation = OrthogonalTransformation(
            x=(2, 0, 0), z=(0, 0, 3))
        vectors = [
            (1, 0, 0),
            (1, 1, 0),
            (1, 1, 1)
        ]
        assert_allclose(transformation.pushforward(vectors), vectors)

    def test_Pushforward_GivenYZUnitBasis_ReturnOriginalVectors(self):
        transformation = OrthogonalTransformation(
            y=(0, 1, 0), z=(0, 0, 1))
        vectors = [
            (1, 0, 0),
            (1, 1, 0),
            (1, 1, 1)
        ]
        assert_allclose(transformation.pushforward(vectors), vectors)

    def test_Pushforward_GivenYZNonUnitBasis_ReturnOriginalVectors(self):
        transformation = OrthogonalTransformation(
            y=(0, 2, 0), z=(0, 0, 3))
        vectors = [
            (1, 0, 0),
            (1, 1, 0),
            (1, 1, 1)
        ]
        assert_allclose(transformation.pushforward(vectors), vectors)

    def test_Pushforward_GivenNoBasisVectors_RaiseRuntimeError(self):
        with pytest.raises(RuntimeError):
            OrthogonalTransformation()

    def test_Pushforward_GivenNonOrthogonalBasis_RaiseRuntimeError(self):
        with pytest.raises(RuntimeError):
            OrthogonalTransformation(x=(1, 0, 0), y=(0, 1, 0), z=(0, 1, 1))

    def test_Pushforward_GivenNonOrthogonalXYBasis_RaiseRuntimeError(self):
        with pytest.raises(RuntimeError):
            OrthogonalTransformation(x=(1, 0, 0), y=(1, 1, 0))

    def test_Pushforward_GivenNonOrthogonalXZBasis_RaiseRuntimeError(self):
        with pytest.raises(RuntimeError):
            OrthogonalTransformation(x=(1, 0, 0), z=(1, 0, 1))

    def test_Pushforward_GivenNonOrthogonalYZBasis_RaiseRuntimeError(self):
        with pytest.raises(RuntimeError):
            OrthogonalTransformation(y=(0, 1, 0), z=(0, 1, 1))

    def test_Pullback_GivenUnitBasis_ReturnOriginalVectors(self):
        transformation = OrthogonalTransformation(
            x=(1, 0, 0), y=(0, 1, 0), z=(0, 0, 1))
        vectors = [
            (1, 0, 0),
            (1, 1, 0),
            (1, 1, 1)
        ]
        assert_allclose(transformation.pullback(vectors), vectors)

    def test_Pullback_GivenRotatedBasis_ReturnRotatedVectors(self):
        transformation = OrthogonalTransformation(
            x=(0, 1, 0), y=(-1, 0, 0), z=(0, 0, 1))
        vectors = [
            (0, 1, 0),
            (-1, 1, 0),
            (-1, 1, 1)
        ]
        expected_vectors = [
            (1, 0, 0),
            (1, 1, 0),
            (1, 1, 1)
        ]
        assert_allclose(transformation.pullback(vectors), expected_vectors)

    def test_PullBack_ComplexExample_ReturnCorrectVectors(self):
        transformation = OrthogonalTransformation(
            x=(-2, 0, 2), y=(-1, 2, -1), z=(-1, -1, -1))
        vectors = (
            (+0, +0, +0),
            (-2, +0, +2),
            (-2, -2, +4),
            (+0, -6, +6),
            (+2, -8, +6),
            (+4, -8, +4),
            (+4, -6, +2),
            (+2, -2, +0)
        )
        expected_vectors = (
            (+0, +0, +0),
            (+2*sqrt(2), +0, +0),
            (+3*sqrt(2), -1*sqrt(6), +0),
            (+3*sqrt(2), -3*sqrt(6), +0),
            (+2*sqrt(2), -4*sqrt(6), +0),
            (+0, -4*sqrt(6), +0),
            (-1*sqrt(2), -3*sqrt(6), +0),
            (-1*sqrt(2), -1*sqrt(6), +0)
        )
        assert_allclose(
            transformation.pullback(vectors), expected_vectors, atol=1e-8)
