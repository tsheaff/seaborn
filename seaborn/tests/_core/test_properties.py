
import numpy as np
import pandas as pd
from matplotlib.colors import same_color

import pytest

from seaborn._core.rules import categorical_order
from seaborn._core.scales import Nominal
from seaborn._core.properties import (
    Color,
)
from seaborn.palettes import color_palette


class TestColor:

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

    @pytest.mark.xfail(reason="Need decision on new behavior")
    def test_nominal_list_too_long(self, cat_vector, cat_order):

        n = len(cat_order) + 1
        palette = color_palette("Oranges", n)
        msg = rf"The edgecolor list has more values \({n}\) than needed \({n - 1}\)"
        with pytest.warns(UserWarning, match=msg):
            Color("edgecolor").get_mapping(Nominal(palette), cat_vector)

    def test_inference_list_arg_numeric_data(self, num_vector):

        palette = color_palette("Reds", 4)
        scale = Color().infer_scale(palette, num_vector)
        assert isinstance(scale, Nominal)
