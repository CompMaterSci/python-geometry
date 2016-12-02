# -*- coding: utf-8 -*-
from python_geometry.simple_polygon import SimplePolygon


class TestEquality(object):

    def test_SelfPolygon_ReturnTrue(self):
        polygon = SimplePolygon(
            vertices=[
                (0, 0, 0),
                (1, 0, 0),
                (1, 1, 0),
                (0, 1, 0)
            ])
        assert polygon == polygon

    def test_TwoIdenticalPolygons_ReturnTrue(self):
        polygon_0 = SimplePolygon(
            vertices=[
                (0, 0, 0),
                (1, 0, 0),
                (1, 1, 0),
                (0, 1, 0)
            ])
        polygon_1 = SimplePolygon(
            vertices=[
                (0, 0, 0),
                (1, 0, 0),
                (1, 1, 0),
                (0, 1, 0)
            ])
        assert polygon_0 == polygon_1

    def test_TwoEqualPolygonsWithDifferentVertexOrder_ReturnTrue(self):
        polygon_0 = SimplePolygon(
            vertices=[
                (0, 0, 0),
                (1, 0, 0),
                (1, 1, 0),
                (0, 1, 0)
            ])
        polygon_1 = SimplePolygon(
            vertices=[
                (1, 1, 0),
                (0, 1, 0),
                (0, 0, 0),
                (1, 0, 0)
            ])
        assert polygon_0 == polygon_1

    def test_TwoUnequalPolygonsWithCommonVertices_ReturnFalse(self):
        polygon_0 = SimplePolygon(
            vertices=[
                (0, 0, 0),
                (1, 0, 0),
                (1, 1, 0),
                (0, 1, 0)
            ])
        polygon_1 = SimplePolygon(
            vertices=[
                (0, 0, 0),
                (1, 0, 0),
                (0, 1, 0)
            ])
        assert not polygon_0 == polygon_1

    def test_TwoUnequalPolygonsWithoutCommonVertices_ReturnFalse(self):
        polygon_0 = SimplePolygon(
            vertices=[
                (0, 0, 0),
                (1, 0, 0),
                (1, 1, 0),
                (0, 1, 0)
            ])
        polygon_1 = SimplePolygon(
            vertices=[
                (0, 0, 1),
                (1, 0, 1),
                (1, 1, 1),
                (0, 1, 1)
            ])
        assert not polygon_0 == polygon_1

