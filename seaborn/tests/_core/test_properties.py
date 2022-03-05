
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.colors import same_color

import pytest

from seaborn._core.rules import categorical_order
from seaborn._core.scales import Nominal, Continuous
from seaborn._core.properties import (
    Color,
    LineStyle,
    Marker,
)
from seaborn._compat import MarkerStyle
from seaborn.palettes import color_palette


class DataFixtures:

    @pytest.fixture
    def num_vector(self, long_df):
        return long_df["s"]

    @pytest.fixture
    def num_order(self, num_vector):
        return categorical_order(num_vector)

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

    @pytest.fixture
    def vectors(self, num_vector, cat_vector):
        return {"num": num_vector, "cat": cat_vector}


class TestColor(DataFixtures):

    def test_nominal_default_palette(self, cat_vector, cat_order):

        m = Color().get_mapping(Nominal(), cat_vector)
        n = len(cat_order)
        actual = m(np.arange(n))
        expected = color_palette(None, n)
        for have, want in zip(actual, expected):
            assert same_color(have, want)

    def test_nominal_default_palette_large(self):

        vector = pd.Series(list("abcdefghijklmnopqrstuvwxyz"))
        m = Color().get_mapping(Nominal(), vector)
        actual = m(np.arange(26))
        expected = color_palette("husl", 26)
        for have, want in zip(actual, expected):
            assert same_color(have, want)

    def test_nominal_named_palette(self, cat_vector, cat_order):

        palette = "Blues"
        m = Color().get_mapping(Nominal(palette), cat_vector)
        n = len(cat_order)
        actual = m(np.arange(n))
        expected = color_palette(palette, n)
        for have, want in zip(actual, expected):
            assert same_color(have, want)

    def test_nominal_list_palette(self, cat_vector, cat_order):

        palette = color_palette("Reds", len(cat_order))
        m = Color().get_mapping(Nominal(palette), cat_vector)
        actual = m(np.arange(len(palette)))
        expected = palette
        for have, want in zip(actual, expected):
            assert same_color(have, want)

    def test_nominal_dict_palette(self, cat_vector, cat_order):

        colors = color_palette("Greens")
        palette = dict(zip(cat_order, colors))
        m = Color().get_mapping(Nominal(palette), cat_vector)
        n = len(cat_order)
        actual = m(np.arange(n))
        expected = colors
        for have, want in zip(actual, expected):
            assert same_color(have, want)

    def test_nominal_dict_with_missing_keys(self, cat_vector, cat_order):

        palette = dict(zip(cat_order[1:], color_palette("Purples")))
        with pytest.raises(ValueError, match="No entry in color dict"):
            Color("color").get_mapping(Nominal(palette), cat_vector)

    def test_nominal_list_too_short(self, cat_vector, cat_order):

        n = len(cat_order) - 1
        palette = color_palette("Oranges", n)
        msg = rf"The edgecolor list has fewer values \({n}\) than needed \({n + 1}\)"
        with pytest.warns(UserWarning, match=msg):
            Color("edgecolor").get_mapping(Nominal(palette), cat_vector)

    def test_nominal_list_too_long(self, cat_vector, cat_order):

        n = len(cat_order) + 1
        palette = color_palette("Oranges", n)
        msg = rf"The edgecolor list has more values \({n}\) than needed \({n - 1}\)"
        with pytest.warns(UserWarning, match=msg):
            Color("edgecolor").get_mapping(Nominal(palette), cat_vector)

    @pytest.mark.parametrize(
        "data_type,scale_class",
        [("cat", Nominal), ("num", Continuous)]
    )
    def test_default(self, data_type, scale_class, vectors):

        scale = Color().default_scale(vectors[data_type])
        assert isinstance(scale, scale_class)

    def test_default_numeric_data_category_dtype(self, num_vector):

        scale = Color().default_scale(num_vector.astype("category"))
        assert isinstance(scale, Nominal)

    def test_default_binary_data(self):

        x = pd.Series([0, 0, 1, 0, 1], dtype=int)
        scale = Color().default_scale(x)
        assert isinstance(scale, Nominal)

    # TODO default scales for other types

    @pytest.mark.parametrize(
        "values,data_type,scale_class",
        [
            ("viridis", "cat", Nominal),  # Based on variable type
            ("viridis", "num", Continuous),  # Based on variable type
            ("muted", "num", Nominal),  # Based on qualitative palette
            (["r", "g", "b"], "num", Nominal),  # Based on list palette
            ({2: "r", 4: "g", 8: "b"}, "num", Nominal),  # Based on dict palette
            (("r", "b"), "num", Continuous),  # Based on tuple / variable type
            (("g", "m"), "cat", Nominal),  # Based on tuple / variable type
            (mpl.cm.get_cmap("inferno"), "num", Continuous),  # Based on callable
        ]
    )
    def test_inference(self, values, data_type, scale_class, vectors):

        scale = Color().infer_scale(values, vectors[data_type])
        assert isinstance(scale, scale_class)
        assert scale.values == values

    def test_inference_binary_data(self):

        x = pd.Series([0, 0, 1, 0, 1], dtype=int)
        scale = Color().infer_scale("viridis", x)
        assert isinstance(scale, Nominal)


class ObjectPropertyBase(DataFixtures):

    def assert_equal(self, a, b):

        assert a == b

    def hashable(self, x):
        return x

    @pytest.mark.parametrize("data_type", ["cat", "num"])
    def test_default(self, data_type, vectors):

        scale = self.prop().default_scale(vectors[data_type])
        assert isinstance(scale, Nominal)

    @pytest.mark.parametrize("data_type", ["cat", "num"])
    def test_inference_list(self, data_type, vectors):

        scale = self.prop().infer_scale(self.values, vectors[data_type])
        assert isinstance(scale, Nominal)
        assert scale.values == self.values

    @pytest.mark.parametrize("data_type", ["cat", "num"])
    def test_inference_dict(self, data_type, vectors):

        x = vectors[data_type]
        values = dict(zip(categorical_order(x), self.values))
        scale = self.prop().infer_scale(values, x)
        assert isinstance(scale, Nominal)
        assert scale.values == values

    def test_dict_missing(self, cat_vector):

        levels = categorical_order(cat_vector)
        values = dict(zip(levels, self.values[:-1]))
        scale = Nominal(values)
        name = self.prop.__name__.lower()
        msg = f"No entry in {name} dictionary for {repr(levels[-1])}"
        with pytest.raises(ValueError, match=msg):
            self.prop().get_mapping(scale, cat_vector)

    @pytest.mark.parametrize("data_type", ["cat", "num"])
    def test_mapping_default(self, data_type, vectors):

        x = vectors[data_type]
        mapping = self.prop().get_mapping(Nominal(), x)
        n = x.nunique()
        for i, expected in enumerate(self.prop()._default_values(n)):
            actual, = mapping([i])
            self.assert_equal(actual, expected)

    @pytest.mark.parametrize("data_type", ["cat", "num"])
    def test_mapping_from_list(self, data_type, vectors):

        x = vectors[data_type]
        scale = Nominal(self.values)
        mapping = self.prop().get_mapping(scale, x)
        for i, expected in enumerate(self.standardized_values):
            actual, = mapping([i])
            self.assert_equal(actual, expected)

    @pytest.mark.parametrize("data_type", ["cat", "num"])
    def test_mapping_from_dict(self, data_type, vectors):

        x = vectors[data_type]
        levels = categorical_order(x)
        values = dict(zip(levels, self.values[::-1]))
        standardized_values = dict(zip(levels, self.standardized_values[::-1]))

        scale = Nominal(values)
        mapping = self.prop().get_mapping(scale, x)
        for i, level in enumerate(levels):
            actual, = mapping([i])
            expected = standardized_values[level]
            self.assert_equal(actual, expected)

    def test_mapping_with_null_value(self, cat_vector):

        mapping = self.prop().get_mapping(Nominal(self.values), cat_vector)
        actual = mapping(np.array([0, np.nan, 2]))
        v0, _, v2 = self.standardized_values
        expected = [v0, self.prop.null_value, v2]
        for a, b in zip(actual, expected):
            self.assert_equal(a, b)

    def test_unique_default_large_n(self):

        n = 24
        x = pd.Series(np.arange(n))
        mapping = self.prop().get_mapping(Nominal(), x)
        assert len({self.hashable(x_i) for x_i in mapping(x)}) == n


class TestMarker(ObjectPropertyBase):

    prop = Marker
    values = ["o", (5, 2, 0), MarkerStyle("^")]
    standardized_values = [MarkerStyle(x) for x in values]

    def assert_equal(self, a, b):

        assert a.get_path() == b.get_path()
        assert a.get_joinstyle() == b.get_joinstyle()
        assert a.get_transform().to_values() == b.get_transform().to_values()
        assert a.get_fillstyle() == b.get_fillstyle()

    def hashable(self, x):
        return (
            x.get_path(),
            x.get_joinstyle(),
            x.get_transform().to_values(),
            x.get_fillstyle(),
        )


class TestLineStyle(ObjectPropertyBase):

    prop = LineStyle
    values = ["solid", "--", (1, .5)]
    standardized_values = [LineStyle._get_dash_pattern(x) for x in values]
