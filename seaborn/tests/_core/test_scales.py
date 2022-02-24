
import numpy as np
import pandas as pd
import matplotlib as mpl

import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_series_equal

from seaborn._core.scales import (
    Nominal,
    Continuous,
)
from seaborn._core.properties import (
    SizedProperty,
    Coordinate,
    Color,
)
from seaborn.palettes import color_palette


class TestContinuous:

    @pytest.fixture
    def x(self):
        return pd.Series([1, 3, 9], name="x", dtype=float)

    def test_coordinate_defaults(self, x):

        s = Continuous().setup(x, Coordinate())
        assert_series_equal(s(x), x)
        assert_series_equal(s.invert_transform(x), x)

    def test_coordinate_transform(self, x):

        s = Continuous(transform="log").setup(x, Coordinate())
        assert_series_equal(s(x), np.log10(x))
        assert_series_equal(s.invert_transform(s(x)), x)

    def test_coordinate_transform_with_parameter(self, x):

        s = Continuous(transform="pow3").setup(x, Coordinate())
        assert_series_equal(s(x), np.power(x, 3))
        assert_series_equal(s.invert_transform(s(x)), x)

    def test_sized_defaults(self, x):

        s = Continuous().setup(x, SizedProperty())
        assert_array_equal(s(x), [0, .25, 1])
        # TODO assert_series_equal(s.invert_transform(s(x)), x)

    def test_sized_with_range(self, x):

        s = Continuous((1, 3)).setup(x, SizedProperty())
        assert_array_equal(s(x), [1, 1.5, 3])
        # TODO assert_series_equal(s.invert_transform(s(x)), x)

    def test_sized_with_norm(self, x):

        s = Continuous(norm=(3, 7)).setup(x, SizedProperty())
        assert_array_equal(s(x), [-.5, 0, 1.5])
        # TODO assert_series_equal(s.invert_transform(s(x)), x)

    def test_sized_with_range_norm_and_transform(self, x):

        x = pd.Series([1, 10, 100])
        # TODO param order?
        s = Continuous((2, 3), (10, 100), "log").setup(x, SizedProperty())
        assert_array_equal(s(x), [1, 2, 3])
        # TODO assert_series_equal(s.invert_transform(s(x)), x)

    def test_color_defaults(self, x):

        cmap = color_palette("ch:", as_cmap=True)
        s = Continuous().setup(x, Color())
        assert_array_equal(s(x), cmap([0, .25, 1])[:, :3])  # FIXME RGBA

    def test_color_with_named_range(self, x):

        cmap = color_palette("viridis", as_cmap=True)
        s = Continuous("viridis").setup(x, Color())
        assert_array_equal(s(x), cmap([0, .25, 1])[:, :3])  # FIXME RGBA

    def test_color_with_tuple_range(self, x):

        cmap = color_palette("blend:b,g", as_cmap=True)
        s = Continuous(("b", "g")).setup(x, Color())
        assert_array_equal(s(x), cmap([0, .25, 1])[:, :3])  # FIXME RGBA

    def test_color_with_norm(self, x):

        cmap = color_palette("ch:", as_cmap=True)
        s = Continuous(norm=(3, 7)).setup(x, Color())
        assert_array_equal(s(x), cmap([-.5, 0, 1.5])[:, :3])  # FIXME RGBA

    def test_color_with_transform(self, x):

        x = pd.Series([1, 10, 100], name="x", dtype=float)
        cmap = color_palette("ch:", as_cmap=True)
        s = Continuous(transform="log").setup(x, Color())
        assert_array_equal(s(x), cmap([0, .5, 1])[:, :3])  # FIXME RGBA


class TestNominal:

    @pytest.fixture
    def x(self):
        return pd.Series(["a", "c", "b", "c"], name="x")

    def test_coordinate_defaults(self, x):

        s = Nominal().setup(x, Coordinate())
        assert_array_equal(s(x), np.array([0, 1, 2, 1], float))
        assert_array_equal(s.invert_transform(s(x)), s(x))

    def test_coordinate_with_order(self, x):

        s = Nominal(order=["a", "b", "c"]).setup(x, Coordinate())
        assert_array_equal(s(x), np.array([0, 2, 1, 2], float))
        assert_array_equal(s.invert_transform(s(x)), s(x))

    def test_coordinate_with_subset_order(self, x):

        s = Nominal(order=["c", "a"]).setup(x, Coordinate())
        assert_array_equal(s(x), np.array([1, 0, np.nan, 0], float))
        assert_array_equal(s.invert_transform(s(x)), s(x))

    def test_coordinate_axis(self, x):

        ax = mpl.figure.Figure().subplots()
        s = Nominal().setup(x, Coordinate(), ax.xaxis)
        assert_array_equal(s(x), np.array([0, 1, 2, 1], float))
        f = ax.xaxis.get_major_formatter()
        assert f.format_ticks([0, 1, 2]) == ["a", "c", "b"]

    def test_coordinate_axis_with_order(self, x):

        order = ["a", "b", "c"]
        ax = mpl.figure.Figure().subplots()
        s = Nominal(order=order).setup(x, Coordinate(), ax.xaxis)
        assert_array_equal(s(x), np.array([0, 2, 1, 2], float))
        f = ax.xaxis.get_major_formatter()
        assert f.format_ticks([0, 1, 2]) == order

    def test_coordinate_axis_with_subset_order(self, x):

        order = ["c", "a"]
        ax = mpl.figure.Figure().subplots()
        s = Nominal(order=order).setup(x, Coordinate(), ax.xaxis)
        assert_array_equal(s(x), np.array([1, 0, np.nan, 0], float))
        f = ax.xaxis.get_major_formatter()
        assert f.format_ticks([0, 1, 2]) == [*order, ""]

    def test_color_defaults(self, x):

        s = Nominal().setup(x, Color())
        cs = color_palette()
        assert_array_equal(s(x), [cs[0], cs[1], cs[2], cs[1]])

    def test_color_named_palette(self, x):

        pal = "flare"
        s = Nominal(pal).setup(x, Color())
        cs = color_palette(pal, 3)
        assert_array_equal(s(x), [cs[0], cs[1], cs[2], cs[1]])

    def test_color_list_palette(self, x):

        cs = color_palette("crest", 3)
        s = Nominal(cs).setup(x, Color())
        assert_array_equal(s(x), [cs[0], cs[1], cs[2], cs[1]])

    def test_color_dict_palette(self, x):

        cs = color_palette("crest", 3)
        pal = dict(zip("bac", cs))
        s = Nominal(pal).setup(x, Color())
        assert_array_equal(s(x), [cs[1], cs[2], cs[0], cs[2]])
