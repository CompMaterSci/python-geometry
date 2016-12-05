# -*- coding: utf-8 -*-
from numpy import array

from python_geometry.line_segment import LineSegment


class TestEquality(object):

    def test_EqualIntegerLineSegment_ReturnTrue_0(self):
        line_segment_0 = LineSegment(
            end_point_0=array([0, 0, 0]),
            end_point_1=array([1, 1, 1]),
            end_point_0_included=True,
            end_point_1_included=True)
        line_segment_1 = LineSegment(
            end_point_0=array([0, 0, 0]),
            end_point_1=array([1, 1, 1]),
            end_point_0_included=True,
            end_point_1_included=True)
        assert line_segment_0 == line_segment_1

    def test_EqualIntegerLineSegment_ReturnTrue_1(self):
        line_segment_0 = LineSegment(
            end_point_0=array([0, 0, 0]),
            end_point_1=array([1, 1, 1]),
            end_point_0_included=True,
            end_point_1_included=True)
        line_segment_1 = LineSegment(
            end_point_0=array([1, 1, 1]),
            end_point_1=array([0, 0, 0]),
            end_point_0_included=True,
            end_point_1_included=True)
        assert line_segment_0 == line_segment_1

    def test_EqualFloatLineSegment_ReturnTrue_0(self):
        line_segment_0 = LineSegment(
            end_point_0=array([0.3, 0.3, 0.3]),
            end_point_1=array([1.3, 1.3, 1.3]),
            end_point_0_included=True,
            end_point_1_included=True)
        line_segment_1 = LineSegment(
            end_point_0=array([0.3, 0.3, 0.3]),
            end_point_1=array([1.3, 1.3, 1.3]),
            end_point_0_included=True,
            end_point_1_included=True)
        assert line_segment_0 == line_segment_1

    def test_EqualFloatLineSegment_ReturnTrue_1(self):
        line_segment_0 = LineSegment(
            end_point_0=array([0.3, 0.3, 0.3]),
            end_point_1=array([1.3, 1.3, 1.3]),
            end_point_0_included=True,
            end_point_1_included=True)
        line_segment_1 = LineSegment(
            end_point_0=array([1.3, 1.3, 1.3]),
            end_point_1=array([0.3, 0.3, 0.3]),
            end_point_0_included=True,
            end_point_1_included=True)
        assert line_segment_0 == line_segment_1

    def test_NonequalIntegerLineSegments_ReturnFalse_0(self):
        line_segment_0 = LineSegment(
            end_point_0=array([0, 0, 0]),
            end_point_1=array([1, 1, 1]),
            end_point_0_included=True,
            end_point_1_included=True)
        line_segment_1 = LineSegment(
            end_point_0=array([0, 0, 0]),
            end_point_1=array([1, 1, 2]),
            end_point_0_included=True,
            end_point_1_included=True)
        assert not line_segment_0 == line_segment_1

    def test_NonequalIntegerLineSegments_ReturnFalse_1(self):
        line_segment_0 = LineSegment(
            end_point_0=array([0, 0, 0]),
            end_point_1=array([1, 1, 1]),
            end_point_0_included=True,
            end_point_1_included=True)
        line_segment_1 = LineSegment(
            end_point_0=array([0, 0, 0]),
            end_point_1=array([1, 1, 1]),
            end_point_0_included=True,
            end_point_1_included=False)
        assert not line_segment_0 == line_segment_1

    def test_NonequalIntegerLineSegments_ReturnFalse_2(self):
        line_segment_0 = LineSegment(
            end_point_0=array([0, 0, 0]),
            end_point_1=array([1, 1, 1]),
            end_point_0_included=True,
            end_point_1_included=True)
        line_segment_1 = LineSegment(
            end_point_0=array([0, 0, 0]),
            end_point_1=array([1, 1, 1]),
            end_point_0_included=False,
            end_point_1_included=True)
        assert not line_segment_0 == line_segment_1

    def test_NonequalIntegerLineSegments_ReturnFalse_3(self):
        line_segment_0 = LineSegment(
            end_point_0=array([0.3, 0.3, 0.3]),
            end_point_1=array([1.3, 1.3, 1.3]),
            end_point_0_included=True,
            end_point_1_included=True)
        line_segment_1 = LineSegment(
            end_point_0=array([0.3, 0.3, 0.3]),
            end_point_1=array([1.3, 1.3, 2.3]),
            end_point_0_included=True,
            end_point_1_included=True)
        assert not line_segment_0 == line_segment_1


class TestRepr(object):

    def test_CompareAgainstItsRepr_ReturnTrue_0(self):
        line_segment = LineSegment(
            end_point_0=array([0, 0, 0]),
            end_point_1=array([1, 1, 1]))
        assert eval(repr(line_segment)) == line_segment

    def test_CompareAgainstItsRepr_ReturnTrue_1(self):
        line_segment = LineSegment(
            end_point_0=array([0, 0, 0]),
            end_point_1=array([1, 1, 1]),
            end_point_0_included=True,
            end_point_1_included=True)
        assert eval(repr(line_segment)) == line_segment

    def test_CompareAgainstItsRepr_ReturnTrue_2(self):
        line_segment = LineSegment(
            end_point_0=array([0, 0, 0]),
            end_point_1=array([1, 1, 1]),
            end_point_0_included=True,
            end_point_1_included=False)
        assert eval(repr(line_segment)) == line_segment

    def test_CompareAgainstItsRepr_ReturnTrue_3(self):
        line_segment = LineSegment(
            end_point_0=array([0, 0, 0]),
            end_point_1=array([1, 1, 1]),
            end_point_0_included=False,
            end_point_1_included=True)
        assert eval(repr(line_segment)) == line_segment

    def test_CompareAgainstItsRepr_ReturnTrue_4(self):
        line_segment = LineSegment(
            end_point_0=array([0, 0, 0]),
            end_point_1=array([1, 1, 1]),
            end_point_0_included=False,
            end_point_1_included=False)
        assert eval(repr(line_segment)) == line_segment
