.. _transformations:

Transformations
===============

Transformations are used to convert vectors from one coordinate system to
another.


Examples
--------

In this example a transformation corresponding to a 90° rotation around the
:math:`z`-axis is initialized.

>>> from geometry.transformations import OrthogonalTransformation
>>> transformation = OrthogonalTransformation(
...     x=np.array([0, 1, 0]),
...     y=np.array([-1, 0, 0]),
...     z=np.array([0, 0, 1]))

This :class:`~geometry.transformations.OrthogonalTransformation` object can now
be used to transform vectors from the current coordinate system to the target
coordinate system

>>> transformation.pushforward(np.array([1, 0, 1]))
array([ 0.,  1.,  1.])

or vice-versa

>>> transformation.pullback(np.array([0, 1, 1]))
array([ 1.,  0.,  1.])
