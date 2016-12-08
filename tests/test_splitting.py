# -*- coding: utf-8 -*-
import pytest
from numpy import array

from python_geometry.plane import Plane
from python_geometry.simple_polygon import SimplePolygon
from python_geometry.splitting import split_by_plane


class TestSplitByPlane:

    def test_ConcavePolygonThatIsNotSplit_ReturnCorrectPolygons(self):
        polygon = SimplePolygon(
            vertices=array([
                (0, 0, 0),
                (1, 0, 0),
                (1, 1, 0),
                (0, 1, 0)
            ]))
        plane = Plane(
            point_in_plane=array([2, 0, 0]),
            normal_vector=array([1, 0, 0]))
        actual = split_by_plane(
            object_to_split=polygon,
            plane=plane)
        expected = [
            SimplePolygon(
                vertices=array([
                    (0, 0, 0),
                    (1, 0, 0),
                    (1, 1, 0),
                    (0, 1, 0)
                ]))
        ]
        assert actual == expected

    def test_ConcavePolygonThatIsSplit_ReturnCorrectPolygons(self):
        polygon = SimplePolygon(
            vertices=array([
                (0, 0, 0),
                (2, 0, 0),
                (2, 2, 0),
                (0, 2, 0)
            ]))
        plane = Plane(
            point_in_plane=array([1, 0, 0]),
            normal_vector=array([1, 0, 0]))
        actual = split_by_plane(
            object_to_split=polygon,
            plane=plane)
        expected = [
            SimplePolygon(
                vertices=array([
                    (1, 2, 0),
                    (0, 2, 0),
                    (0, 0, 0),
                    (1, 0, 0)
                ])),
            SimplePolygon(
                vertices=array([
                    (1, 0, 0),
                    (2, 0, 0),
                    (2, 2, 0),
                    (1, 2, 0)
                ]))
        ]
        assert actual == expected

    def test_ComplexPolygon_ReturnCorrectPolygons_0(self):
        polygon = SimplePolygon(
            vertices=array([
                (0, 0, 0),
                (4, 0, 0),
                (2, 2, 0),
                (4, 4, 0),
                (0, 4, 0)
            ]))
        plane = Plane(
            point_in_plane=array([1, 0, 0]),
            normal_vector=array([1, 0, 0]))
        actual = split_by_plane(
            object_to_split=polygon,
            plane=plane)
        expected = [
            SimplePolygon(
                vertices=array([
                    (1, 4, 0),
                    (0, 4, 0),
                    (0, 0, 0),
                    (1, 0, 0)
                ])),
            SimplePolygon(
                vertices=array([
                    (1, 0, 0),
                    (4, 0, 0),
                    (2, 2, 0),
                    (4, 4, 0),
                    (1, 4, 0)
                ]))
        ]
        assert actual == expected

    def test_ComplexPolygon_ReturnCorrectPolygons_1(self):
        polygon = SimplePolygon(
            vertices=array([
                (0, 0, 0),
                (4, 0, 0),
                (2, 2, 0),
                (4, 4, 0),
                (0, 4, 0)
            ]))
        plane = Plane(
            point_in_plane=array([2, 0, 0]),
            normal_vector=array([1, 0, 0]))
        actual = split_by_plane(
            object_to_split=polygon,
            plane=plane)
        expected = [
            SimplePolygon(
                vertices=array([
                    (2, 4, 0),
                    (0, 4, 0),
                    (0, 0, 0),
                    (2, 0, 0)
                ])),
            SimplePolygon(
                vertices=array([
                    (2, 0, 0),
                    (4, 0, 0),
                    (2, 2, 0)
                ])),
            SimplePolygon(
                vertices=array([
                    (2, 2, 0),
                    (4, 4, 0),
                    (2, 4, 0)
                ]))
        ]
        assert actual == expected

    def test_ComplexPolygon_ReturnCorrectPolygons_2(self):
        polygon = SimplePolygon(
            vertices=array([
                (0, 0, 0),
                (4, 0, 0),
                (2, 2, 0),
                (4, 4, 0),
                (0, 4, 0)
            ]))
        plane = Plane(
            point_in_plane=array([3, 0, 0]),
            normal_vector=array([1, 0, 0]))
        actual = split_by_plane(
            object_to_split=polygon,
            plane=plane)
        expected = [
            SimplePolygon(
                vertices=array([
                    (3, 4, 0),
                    (0, 4, 0),
                    (0, 0, 0),
                    (3, 0, 0),
                    (3, 1, 0),
                    (2, 2, 0),
                    (3, 3, 0)
                ])),
            SimplePolygon(
                vertices=array([
                    (3, 0, 0),
                    (4, 0, 0),
                    (3, 1, 0)
                ])),
            SimplePolygon(
                vertices=array([
                    (3, 3, 0),
                    (4, 4, 0),
                    (3, 4, 0)
                ]))
        ]
        assert actual == expected

    @pytest.mark.parametrize(
        ('point_in_plane', 'normal_vector'),
        [
            (array([0, 0, 0]), array([1, 0, 0])),
            (array([0, 0, 0]), array([-1, 0, 0])),
            (array([0, 0, 0]), array([0, 1, 0])),
            (array([0, 0, 0]), array([0, -1, 0])),
            (array([1, 1, 0]), array([1, 0, 0])),
            (array([1, 1, 0]), array([-1, 0, 0])),
            (array([1, 1, 0]), array([0, 1, 0])),
            (array([1, 1, 0]), array([0, -1, 0])),
        ]
    )
    def test_SimplePolygonWithSegmentOnPlane_ReturnCorrectPolygon(self, point_in_plane, normal_vector):
        polygon = SimplePolygon(
            vertices=array([
                (0, 0, 0),
                (1, 0, 0),
                (1, 1, 0),
                (0, 1, 0)
            ]))
        plane = Plane(
            point_in_plane=point_in_plane,
            normal_vector=normal_vector)
        actual = split_by_plane(
            object_to_split=polygon,
            plane=plane)
        expected = [
            SimplePolygon(
                vertices=array([
                    (0, 0, 0),
                    (1, 0, 0),
                    (1, 1, 0),
                    (0, 1, 0)
                ]))
        ]
        assert actual == expected

    @pytest.mark.parametrize(
        'normal_vector',
        [
            array([0, 1, 0]),
            array([0, -1, 0])
        ])
    def test_FirstQuadrantLShapedPolygonWithShortEdgeOnPlane_ReturnCorrectPolygons(self, normal_vector):
        polygon = SimplePolygon(
            vertices=array([
                (-1, -1, 0),
                (1, -1, 0),
                (1, 0, 0),
                (0, 0, 0),
                (0, 1, 0),
                (-1, 1, 0)
            ]))
        plane = Plane(
            point_in_plane=array([0, 0, 0]),
            normal_vector=normal_vector)
        actual = split_by_plane(
            object_to_split=polygon,
            plane=plane)
        expected = [
            SimplePolygon(
                vertices=array([
                    (-1, -1, 0),
                    (1, -1, 0),
                    (1, 0, 0),
                    (-1, 0, 0)
                ])),
            SimplePolygon(
                vertices=array([
                    (0, 0, 0),
                    (0, 1, 0),
                    (-1, 1, 0),
                    (-1, 0, 0)
                ]))
        ]
        assert actual == expected

    @pytest.mark.parametrize(
        'normal_vector',
        [
            array([0, 1, 0]),
            array([0, -1, 0])
        ])
    def test_SecondQuadrantLShapedPolygonWithShortEdgeOnPlane_ReturnCorrectPolygons(self, normal_vector):
        polygon = SimplePolygon(
            vertices=array([
                (-1, -1, 0),
                (1, -1, 0),
                (1, 1, 0),
                (0, 1, 0),
                (0, 0, 0),
                (-1, 0, 0)
            ]))
        plane = Plane(
            point_in_plane=array([0, 0, 0]),
            normal_vector=normal_vector)
        actual = split_by_plane(
            object_to_split=polygon,
            plane=plane)
        expected = [
            SimplePolygon(
                vertices=array([
                    (-1, -1, 0),
                    (1, -1, 0),
                    (1, 0, 0),
                    (-1, 0, 0)
                ])),
            SimplePolygon(
                vertices=array([
                    (1, 0, 0),
                    (1, 1, 0),
                    (0, 1, 0),
                    (0, 0, 0)
                ]))
        ]
        assert actual == expected

    @pytest.mark.parametrize(
        'normal_vector',
        [
            array([0, 1, 0]),
            array([0, -1, 0])
        ])
    def test_ThirdQuadrantLShapedPolygonWithShortEdgeOnPlane_ReturnCorrectPolygons(self, normal_vector):
        polygon = SimplePolygon(
            vertices=array([
                (0, -1, 0),
                (1, -1, 0),
                (1, 1, 0),
                (-1, 1, 0),
                (-1, 0, 0),
                (0, 0, 0)
            ]))
        plane = Plane(
            point_in_plane=array([0, 0, 0]),
            normal_vector=normal_vector)
        actual = split_by_plane(
            object_to_split=polygon,
            plane=plane)
        expected = [
            SimplePolygon(
                vertices=array([
                    (0, -1, 0),
                    (1, -1, 0),
                    (1, 0, 0),
                    (0, 0, 0)
                ])),
            SimplePolygon(
                vertices=array([
                    (1, 0, 0),
                    (1, 1, 0),
                    (-1, 1, 0),
                    (-1, 0, 0)
                ]))
        ]
        assert actual == expected

    @pytest.mark.parametrize(
        'normal_vector',
        [
            array([0, 1, 0]),
            array([0, -1, 0])
        ])
    def test_FourthQuadrantLShapedPolygonWithShortEdgeOnPlane_ReturnCorrectPolygons(self, normal_vector):
        polygon = SimplePolygon(
            vertices=array([
                (-1, -1, 0),
                (0, -1, 0),
                (0, 0, 0),
                (1, 0, 0),
                (1, 1, 0),
                (-1, 1, 0)
            ]))
        plane = Plane(
            point_in_plane=array([0, 0, 0]),
            normal_vector=normal_vector)
        actual = split_by_plane(
            object_to_split=polygon,
            plane=plane)
        expected = [
            SimplePolygon(
                vertices=array([
                    (-1, -1, 0),
                    (0, -1, 0),
                    (0, 0, 0),
                    (-1, 0, 0)
                ])),
            SimplePolygon(
                vertices=array([
                    (1, 0, 0),
                    (1, 1, 0),
                    (-1, 1, 0),
                    (-1, 0, 0)
                ]))
        ]
        assert actual == expected
