import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.scale import LinearScale
from matplotlib.colors import Normalize, same_color

import pytest
from numpy.testing import assert_array_equal

from seaborn._core.rules import categorical_order
from seaborn._core.scales_take1 import (
    DateTimeScale,
    NumericScale,
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


class ContinuousBase(MappingsBase):

    @staticmethod
    def norm(x, vmin, vmax):
        normed = x - vmin
        normed /= vmax - vmin
        return normed

    @staticmethod
    def transform(x, lo, hi):
        return lo + x * (hi - lo)

    def test_default_numeric(self):

        x = pd.Series([-1, .4, 2, 1.2])
        scale = self.default_scale(x)
        y = self.semantic().setup(x, scale)(x)
        normed = self.norm(x, x.min(), x.max())
        expected = self.transform(normed, *self.semantic().default_range)
        assert_array_equal(y, expected)

    def test_default_categorical(self):

        x = pd.Series(["a", "c", "b", "c"])
        scale = self.default_scale(x)
        y = self.semantic().setup(x, scale)(x)
        normed = np.array([1, .5, 0, .5])
        expected = self.transform(normed, *self.semantic().default_range)
        assert_array_equal(y, expected)

    def test_range_numeric(self):

        values = (1, 5)
        x = pd.Series([-1, .4, 2, 1.2])
        scale = self.default_scale(x)
        y = self.semantic(values).setup(x, scale)(x)
        normed = self.norm(x, x.min(), x.max())
        expected = self.transform(normed, *values)
        assert_array_equal(y, expected)

    def test_range_categorical(self):

        values = (1, 5)
        x = pd.Series(["a", "c", "b", "c"])
        scale = self.default_scale(x)
        y = self.semantic(values).setup(x, scale)(x)
        normed = np.array([1, .5, 0, .5])
        expected = self.transform(normed, *values)
        assert_array_equal(y, expected)

    def test_list_numeric(self):

        values = [.3, .8, .5]
        x = pd.Series([2, 500, 10, 500])
        expected = [.3, .5, .8, .5]
        scale = self.default_scale(x)
        y = self.semantic(values).setup(x, scale)(x)
        assert_array_equal(y, expected)

    def test_list_categorical(self):

        values = [.2, .6, .4]
        x = pd.Series(["a", "c", "b", "c"])
        expected = [.2, .6, .4, .6]
        scale = self.default_scale(x)
        y = self.semantic(values).setup(x, scale)(x)
        assert_array_equal(y, expected)

    def test_list_implies_categorical(self):

        x = pd.Series([2, 500, 10, 500])
        values = [.2, .6, .4]
        expected = [.2, .4, .6, .4]
        scale = self.default_scale(x)
        y = self.semantic(values).setup(x, scale)(x)
        assert_array_equal(y, expected)

    def test_dict_numeric(self):

        x = pd.Series([2, 500, 10, 500])
        values = {2: .3, 500: .5, 10: .8}
        scale = self.default_scale(x)
        y = self.semantic(values).setup(x, scale)(x)
        assert_array_equal(y, x.map(values))

    def test_dict_categorical(self):

        x = pd.Series(["a", "c", "b", "c"])
        values = {"a": .3, "b": .5, "c": .8}
        scale = self.default_scale(x)
        y = self.semantic(values).setup(x, scale)(x)
        assert_array_equal(y, x.map(values))

    def test_norm_numeric(self):

        x = pd.Series([2, 500, 10])
        norm = mpl.colors.LogNorm(1, 100)
        scale = NumericScale(LinearScale("x"), norm=norm)
        y = self.semantic().setup(x, scale)(x)
        x = np.asarray(x)  # matplotlib<3.4.3 compatability
        expected = self.transform(norm(x), *self.semantic().default_range)
        assert_array_equal(y, expected)

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
