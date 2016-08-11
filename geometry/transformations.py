# -*- coding: utf-8 -*-
"""
    This module implements coordinate transformations.

    .. versionadded:: 0.1

"""
from numpy import allclose, asanyarray, cross, dot, einsum, isclose
from numpy.linalg import inv, norm


def _build_row_vector_matrix(*vectors):
    # Copy makes sure that the data is contiguous in memory
    return asanyarray(vectors)


def _are_orthogonal(vector_0, vector_1):
    return bool(isclose(dot(vector_0, vector_1), 0))


class AffineTransformation(object):
    """The affine transformation object implements the basic interface to
    affine coordinate transformations.

    Parameters
    ----------
    x, y, z
        The new basis vectors expressed in the terms of the reference basis.

    """

    def __init__(self, x, y, z):
        self._x = x
        self._y = y
        self._z = z
        self.transformation_matrix = self._build_transformation_matrix(x, y, z)
        self.inverse_transformation_matrix = (
            self._build_inverse_transformation_matrix(
                self.transformation_matrix))

    @staticmethod
    def _build_transformation_matrix(x, y, z):
        return _build_row_vector_matrix(x, y, z)

    @staticmethod
    def _build_inverse_transformation_matrix(transformation_matrix):
        return inv(transformation_matrix)

    def pushforward(self, vectors):
        """Transform vectors from reference to target coordinate system.

        Parameters
        ----------
        vectors
            Vectors in the reference coordinate system.

        Returns
        -------
        np.ndarray
            Vectors in the target coordinate system.

        """
        v = asanyarray(vectors)
        return einsum('...i,ij->...j', v, self.transformation_matrix)

    def pullback(self, vectors):
        """Transform vectors from target to reference coordinate system.

        Parameters
        ----------
        vectors
            Vectors in the target coordinate system.

        Returns
        -------
        np.ndarray
            Vectors in the reference coordinate system.

        """
        v = asanyarray(vectors)
        return einsum('...i,ij->...j', v, self.inverse_transformation_matrix)

    def __repr__(self):
        return (
            '{self.__class__.__name__}('
            'x={self._x!r}, '
            'y={self._y!r}, '
            'z={self._z!r}'
            ')').format(self=self)


class OrthogonalTransformation(AffineTransformation):
    """Transform coordinates using an orthogonal transformation.

    Parameters
    ----------
    x, y, z
        The new basis vectors expressed in the terms of the reference basis. At
        least two have to be given. If all three are given a check is performed
        to make sure that the new basis is orthogonal and right-handed.

    """

    def __init__(self, x=None, y=None, z=None):
        if x is not None:
            x = asanyarray(x)
        if y is not None:
            y = asanyarray(y)
        if z is not None:
            z = asanyarray(z)

        all_new_basis_vectors_given = False
        if x is not None and y is not None and z is not None:
            all_new_basis_vectors_given = True
        elif x is not None and y is not None and z is None:
            if not _are_orthogonal(x, y):
                raise RuntimeError(
                    'Non orthogonal target basis vectors given.')
            z = cross(x, y)
        elif x is not None and y is None and z is not None:
            if not _are_orthogonal(x, z):
                raise RuntimeError(
                    'Non orthogonal target basis vectors given.')
            y = cross(z, x)
        elif x is None and y is not None and z is not None:
            if not _are_orthogonal(y, z):
                raise RuntimeError(
                    'Non orthogonal target basis vectors given.')
            x = cross(y, z)
        else:
            raise RuntimeError('Not enough target basis vectors given.')
        x = x/norm(x)
        y = y/norm(y)
        z = z/norm(z)
        if all_new_basis_vectors_given and not allclose(cross(x, y), z):
            raise RuntimeError(
                'Unable to build transformation with given target basis, as '
                'they are non-orthogonal.')

        super(OrthogonalTransformation, self).__init__(x, y, z)

    @staticmethod
    def _build_inverse_transformation_matrix(transformation_matrix):
        # The inverse of an orthonormal matrix
        # Copy makes sure that the data is contiguous in memory
        return transformation_matrix.T.copy()
