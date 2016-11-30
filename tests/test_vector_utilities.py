# -*- coding: utf-8 -*-
import pytest

from python_geometry.vector_utilities import are_parallel


class TestAreParallel:

    @pytest.mark.parametrize(
        ('a', 'b', 'expected'),
        [
            ((1, 0, 0), (1, 0, 0), True),
            ((1, 0, 0), (2, 0, 0), True),
            ((1, 0, 0), (-1, 0, 0), True),
            ((1, 0, 0), (-2, 0, 0), True),
            ((1, 0, 0), (0, 0, 0), False),
            ((1, 0, 0), (1, 1, 0), False)
        ]
    )
    def test_GivenVectors_ReturnAreParallel(self, a, b, expected):
        assert are_parallel(a, b) is expected
