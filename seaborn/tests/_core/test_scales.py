
import numpy as np
import pandas as pd
import matplotlib as mpl

import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_series_equal

from seaborn._core.scales import (
    Continuous,
)
from seaborn._core.properties import (
    Coordinate,
    SizedProperty,
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
