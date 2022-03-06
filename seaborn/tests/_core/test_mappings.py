import numpy as np
import pandas as pd
from matplotlib.scale import LinearScale
from matplotlib.colors import Normalize, same_color

import pytest
from numpy.testing import assert_array_equal

from seaborn._core.rules import categorical_order
from seaborn._core.scales_take1 import (
    DateTimeScale,
    get_default_scale,
)
from seaborn._core.mappings import (
    ColorSemantic,
    WidthSemantic,
    EdgeWidthSemantic,
    LineWidthSemantic,
)
from seaborn.palettes import color_palette


class MappingsBase:

    def default_scale(self, data):
        return get_default_scale(data).setup(data)


class TestColor(MappingsBase):

    @pytest.fixture
    def num_vector(self, long_df):
        return long_df["s"]

    @pytest.fixture
    def num_order(self, num_vector):
        return categorical_order(num_vector)

    @pytest.fixture
    def num_scale(self, num_vector):
        norm = Normalize()
        norm.autoscale(num_vector)
        scale = get_default_scale(num_vector)
        return scale

    @pytest.fixture
    def cat_vector(self, long_df):
        return long_df["a"]

    @pytest.fixture
    def cat_order(self, cat_vector):
        return categorical_order(cat_vector)

    @pytest.fixture
    def dt_num_vector(self, long_df):
        return long_df["t"]

    @pytest.fixture
    def dt_cat_vector(self, long_df):
        return long_df["d"]

    def test_datetime_default_palette(self, dt_num_vector):

        scale = self.default_scale(dt_num_vector)
        m = ColorSemantic().setup(dt_num_vector, scale)
        mapped = m(dt_num_vector)

        tmp = dt_num_vector - dt_num_vector.min()
        normed = tmp / tmp.max()

        expected_cmap = color_palette("ch:", as_cmap=True)
        expected = expected_cmap(normed)

        assert len(mapped) == len(expected)
        for have, want in zip(mapped, expected):
            assert same_color(have, want)

    def test_datetime_specified_palette(self, dt_num_vector):

        palette = "mako"
        scale = self.default_scale(dt_num_vector)
        m = ColorSemantic(palette=palette).setup(dt_num_vector, scale)
        mapped = m(dt_num_vector)

        tmp = dt_num_vector - dt_num_vector.min()
        normed = tmp / tmp.max()

        expected_cmap = color_palette(palette, as_cmap=True)
        expected = expected_cmap(normed)

        assert len(mapped) == len(expected)
        for have, want in zip(mapped, expected):
            assert same_color(have, want)

    def test_datetime_norm_limits(self, dt_num_vector):

        norm = (
            dt_num_vector.min() - pd.Timedelta(2, "m"),
            dt_num_vector.max() - pd.Timedelta(1, "m"),
        )
        palette = "mako"

        scale = DateTimeScale(LinearScale("color"), norm=norm)
        m = ColorSemantic(palette=palette).setup(dt_num_vector, scale)
        mapped = m(dt_num_vector)

        tmp = dt_num_vector - norm[0]
        normed = tmp / (norm[1] - norm[0])

        expected_cmap = color_palette(palette, as_cmap=True)
        expected = expected_cmap(normed)

        assert len(mapped) == len(expected)
        for have, want in zip(mapped, expected):
            assert same_color(have, want)


class ContinuousBase:

    def test_default_datetime(self):

        x = pd.Series(np.array([10000, 10100, 10101], dtype="datetime64[D]"))
        scale = self.default_scale(x)
        y = self.semantic().setup(x, scale)(x)
        tmp = x - x.min()
        normed = tmp / tmp.max()
        expected = self.transform(normed, *self.semantic().default_range)
        assert_array_equal(y, expected)

    def test_range_datetime(self):

        values = .2, .9
        x = pd.Series(np.array([10000, 10100, 10101], dtype="datetime64[D]"))
        scale = self.default_scale(x)
        y = self.semantic(values).setup(x, scale)(x)
        tmp = x - x.min()
        normed = tmp / tmp.max()
        expected = self.transform(normed, *values)
        assert_array_equal(y, expected)


class TestWidth(ContinuousBase):

    semantic = WidthSemantic


class TestLineWidth(ContinuousBase):

    semantic = LineWidthSemantic


class TestEdgeWidth(ContinuousBase):

    semantic = EdgeWidthSemantic
