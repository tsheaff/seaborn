import functools
import itertools
import warnings
import imghdr
from distutils.version import LooseVersion

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from numpy.testing import assert_array_equal

from seaborn._core.plot import Plot
from seaborn._core.scales import Nominal, Continuous
from seaborn._core.rules import categorical_order
from seaborn._core.moves import Move
from seaborn._marks.base import Mark
from seaborn._stats.base import Stat

assert_vector_equal = functools.partial(assert_series_equal, check_names=False)


def assert_gridspec_shape(ax, nrows=1, ncols=1):

    gs = ax.get_gridspec()
    if LooseVersion(mpl.__version__) < "3.2":
        assert gs._nrows == nrows
        assert gs._ncols == ncols
    else:
        assert gs.nrows == nrows
        assert gs.ncols == ncols


class MockMark(Mark):

    # TODO we need to sort out the stat application, it is broken right now
    # default_stat = MockStat
    grouping_vars = ["color"]

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.passed_keys = []
        self.passed_data = []
        self.passed_axes = []
        self.passed_scales = []
        self.n_splits = 0

    def _plot_split(self, keys, data, ax, kws):

        self.n_splits += 1
        self.passed_keys.append(keys)
        self.passed_data.append(data)
        self.passed_axes.append(ax)

        self.passed_scales.append(self.scales)

    def _legend_artist(self, variables, value):

        a = mpl.lines.Line2D([], [])
        a.variables = variables
        a.value = value
        return a


class TestInit:

    def test_empty(self):

        p = Plot()
        assert p._data.source_data is None
        assert p._data.source_vars == {}

    def test_data_only(self, long_df):

        p = Plot(long_df)
        assert p._data.source_data is long_df
        assert p._data.source_vars == {}

    def test_df_and_named_variables(self, long_df):

        variables = {"x": "a", "y": "z"}
        p = Plot(long_df, **variables)
        for var, col in variables.items():
            assert_vector_equal(p._data.frame[var], long_df[col])
        assert p._data.source_data is long_df
        assert p._data.source_vars.keys() == variables.keys()

    def test_df_and_mixed_variables(self, long_df):

        variables = {"x": "a", "y": long_df["z"]}
        p = Plot(long_df, **variables)
        for var, col in variables.items():
            if isinstance(col, str):
                assert_vector_equal(p._data.frame[var], long_df[col])
            else:
                assert_vector_equal(p._data.frame[var], col)
        assert p._data.source_data is long_df
        assert p._data.source_vars.keys() == variables.keys()

    def test_vector_variables_only(self, long_df):

        variables = {"x": long_df["a"], "y": long_df["z"]}
        p = Plot(**variables)
        for var, col in variables.items():
            assert_vector_equal(p._data.frame[var], col)
        assert p._data.source_data is None
        assert p._data.source_vars.keys() == variables.keys()

    def test_vector_variables_no_index(self, long_df):

        variables = {"x": long_df["a"].to_numpy(), "y": long_df["z"].to_list()}
        p = Plot(**variables)
        for var, col in variables.items():
            assert_vector_equal(p._data.frame[var], pd.Series(col))
            assert p._data.names[var] is None
        assert p._data.source_data is None
        assert p._data.source_vars.keys() == variables.keys()

    def test_data_only_named(self, long_df):

        p = Plot(data=long_df)
        assert p._data.source_data is long_df
        assert p._data.source_vars == {}

    def test_positional_and_named_data(self, long_df):

        err = "`data` given by both name and position"
        with pytest.raises(TypeError, match=err):
            Plot(long_df, data=long_df)

    @pytest.mark.parametrize("var", ["x", "y"])
    def test_positional_and_named_xy(self, long_df, var):

        err = f"`{var}` given by both name and position"
        with pytest.raises(TypeError, match=err):
            Plot(long_df, "a", "b", **{var: "c"})

    def test_positional_data_x_y(self, long_df):

        p = Plot(long_df, "a", "b")
        assert p._data.source_data is long_df
        assert list(p._data.source_vars) == ["x", "y"]

    def test_positional_x_y(self, long_df):

        p = Plot(long_df["a"], long_df["b"])
        assert p._data.source_data is None
        assert list(p._data.source_vars) == ["x", "y"]

    def test_positional_data_x(self, long_df):

        p = Plot(long_df, "a")
        assert p._data.source_data is long_df
        assert list(p._data.source_vars) == ["x"]

    def test_positional_x(self, long_df):

        p = Plot(long_df["a"])
        assert p._data.source_data is None
        assert list(p._data.source_vars) == ["x"]

    def test_positional_too_many(self, long_df):

        err = r"Plot accepts no more than 3 positional arguments \(data, x, y\)"
        with pytest.raises(TypeError, match=err):
            Plot(long_df, "x", "y", "z")


class TestLayerAddition:

    def test_without_data(self, long_df):

        p = Plot(long_df, x="x", y="y").add(MockMark()).plot()
        layer, = p._layers
        assert_frame_equal(p._data.frame, layer["data"].frame)

    def test_with_new_variable_by_name(self, long_df):

        p = Plot(long_df, x="x").add(MockMark(), y="y").plot()
        layer, = p._layers
        assert layer["data"].frame.columns.to_list() == ["x", "y"]
        for var in "xy":
            assert_vector_equal(layer["data"].frame[var], long_df[var])

    def test_with_new_variable_by_vector(self, long_df):

        p = Plot(long_df, x="x").add(MockMark(), y=long_df["y"]).plot()
        layer, = p._layers
        assert layer["data"].frame.columns.to_list() == ["x", "y"]
        for var in "xy":
            assert_vector_equal(layer["data"].frame[var], long_df[var])

    def test_with_late_data_definition(self, long_df):

        p = Plot().add(MockMark(), data=long_df, x="x", y="y").plot()
        layer, = p._layers
        assert layer["data"].frame.columns.to_list() == ["x", "y"]
        for var in "xy":
            assert_vector_equal(layer["data"].frame[var], long_df[var])

    def test_with_new_data_definition(self, long_df):

        long_df_sub = long_df.sample(frac=.5)

        p = Plot(long_df, x="x", y="y").add(MockMark(), data=long_df_sub).plot()
        layer, = p._layers
        assert layer["data"].frame.columns.to_list() == ["x", "y"]
        for var in "xy":
            assert_vector_equal(
                layer["data"].frame[var], long_df_sub[var].reindex(long_df.index)
            )

    def test_drop_variable(self, long_df):

        p = Plot(long_df, x="x", y="y").add(MockMark(), y=None).plot()
        layer, = p._layers
        assert layer["data"].frame.columns.to_list() == ["x"]
        assert_vector_equal(layer["data"].frame["x"], long_df["x"])

    @pytest.mark.xfail(reason="Need decision on default stat")
    def test_stat_default(self):

        class MarkWithDefaultStat(Mark):
            default_stat = Stat

        p = Plot().add(MarkWithDefaultStat())
        layer, = p._layers
        assert layer["stat"].__class__ is Stat

    def test_stat_nondefault(self):

        class MarkWithDefaultStat(Mark):
            default_stat = Stat

        class OtherMockStat(Stat):
            pass

        p = Plot().add(MarkWithDefaultStat(), OtherMockStat())
        layer, = p._layers
        assert layer["stat"].__class__ is OtherMockStat

    @pytest.mark.parametrize(
        "arg,expected",
        [("x", "x"), ("y", "y"), ("v", "x"), ("h", "y")],
    )
    def test_orient(self, arg, expected):

        class MockStatTrackOrient(Stat):
            def __call__(self, data, groupby, orient):
                self.orient_at_call = orient
                return data

        class MockMoveTrackOrient(Move):
            def __call__(self, data, groupby, orient):
                self.orient_at_call = orient
                return data

        s = MockStatTrackOrient()
        m = MockMoveTrackOrient()
        Plot(x=[1, 2, 3], y=[1, 2, 3]).add(MockMark(), s, m, orient=arg).plot()

        assert s.orient_at_call == expected
        assert m.orient_at_call == expected


class TestAxisScaling:

    @pytest.mark.xfail(reason="Calendric scale not implemented")
    def test_inference(self, long_df):

        for col, scale_type in zip("zat", ["continuous", "nominal", "calendric"]):
            p = Plot(long_df, x=col, y=col).add(MockMark()).plot()
            for var in "xy":
                assert p._scales[var].scale_type == scale_type

    def test_inference_from_layer_data(self):

        p = Plot().add(MockMark(), x=["a", "b", "c"]).plot()
        assert p._scales["x"]("b") == 1

    def test_inference_concatenates(self):

        p = Plot(x=[1, 2, 3]).add(MockMark(), x=["a", "b", "c"]).plot()
        assert p._scales["x"]("b") == 4

    def test_inferred_categorical_converter(self):

        p = Plot(x=["b", "c", "a"]).add(MockMark()).plot()
        ax = p._figure.axes[0]
        assert ax.xaxis.convert_units("c") == 1

    def test_explicit_categorical_converter(self):

        p = Plot(y=[2, 1, 3]).scale(y=Nominal()).add(MockMark()).plot()
        ax = p._figure.axes[0]
        assert ax.yaxis.convert_units("3") == 2

    def test_categorical_as_datetime(self):

        dates = ["1970-01-03", "1970-01-02", "1970-01-04"]
        p = Plot(x=dates).scale_datetime("x").add(MockMark()).plot()
        ax = p._figure.axes[0]
        assert ax.xaxis.converter

    @pytest.mark.xfail(reason="Custom log scale needs log name for consistency")
    def test_faceted_log_scale(self):

        p = Plot(y=[1, 10]).facet(col=["a", "b"]).scale(y="log").plot()
        for ax in p._figure.axes:
            assert ax.get_yscale() == "log"

    @pytest.mark.xfail(reason="Custom log scale needs log name for consistency")
    def test_faceted_log_scale_without_data(self):

        p = Plot(y=[1, 10]).facet(col=["a", "b"]).scale(y="log").plot()
        for ax in p._figure.axes:
            assert ax.get_yscale() == "log"

    @pytest.mark.xfail(reason="Custom log scale needs log name for consistency")
    def test_paired_single_log_scale(self):

        x0, x1 = [1, 2, 3], [1, 10, 100]
        p = Plot().pair(x=[x0, x1]).scale(x1="log").plot()
        ax0, ax1 = p._figure.axes
        assert ax0.get_xscale() == "linear"
        assert ax1.get_xscale() == "log"

    def test_mark_data_log_transform_is_inverted(self, long_df):

        col = "z"
        m = MockMark()
        Plot(long_df, x=col).scale(x="log").add(m).plot()
        assert_vector_equal(m.passed_data[0]["x"], long_df[col])

    def test_mark_data_log_transfrom_with_stat(self, long_df):

        class Mean(Stat):
            group_by_orient = True

            def __call__(self, data, groupby, orient):
                other = {"x": "y", "y": "x"}[orient]
                return groupby.agg(data, {other: "mean"})

        col = "z"
        grouper = "a"
        m = MockMark()
        s = Mean()

        Plot(long_df, x=grouper, y=col).scale(y="log").add(m, s).plot()

        expected = (
            long_df[col]
            .pipe(np.log)
            .groupby(long_df[grouper], sort=False)
            .mean()
            .pipe(np.exp)
            .reset_index(drop=True)
        )
        assert_vector_equal(m.passed_data[0]["y"], expected)

    def test_mark_data_from_categorical(self, long_df):

        col = "a"
        m = MockMark()
        Plot(long_df, x=col).add(m).plot()

        levels = categorical_order(long_df[col])
        level_map = {x: float(i) for i, x in enumerate(levels)}
        assert_vector_equal(m.passed_data[0]["x"], long_df[col].map(level_map))

    @pytest.mark.xfail(reason="Calendric scale not implemented yet")
    def test_mark_data_from_datetime(self, long_df):

        col = "t"
        m = MockMark()
        Plot(long_df, x=col).add(m).plot()

        expected = long_df[col].map(mpl.dates.date2num)
        if LooseVersion(mpl.__version__) < "3.3":
            expected = expected + mpl.dates.date2num(np.datetime64('0000-12-31'))

        assert_vector_equal(m.passed_data[0]["x"], expected)

    def test_facet_categories(self):

        m = MockMark()
        p = Plot(x=["a", "b", "a", "c"], col=["x", "x", "y", "y"]).add(m).plot()
        ax1, ax2 = p._figure.axes
        assert len(ax1.get_xticks()) == 3
        assert len(ax2.get_xticks()) == 3
        assert_vector_equal(m.passed_data[0]["x"], pd.Series([0., 1.], [0, 1]))
        assert_vector_equal(m.passed_data[1]["x"], pd.Series([0., 2.], [2, 3]))

    def test_facet_categories_unshared(self):

        m = MockMark()
        p = (
            Plot(x=["a", "b", "a", "c"], col=["x", "x", "y", "y"])
            .configure(sharex=False)
            .add(m)
            .plot()
        )
        ax1, ax2 = p._figure.axes
        assert len(ax1.get_xticks()) == 2
        assert len(ax2.get_xticks()) == 2
        assert_vector_equal(m.passed_data[0]["x"], pd.Series([0., 1.], [0, 1]))
        assert_vector_equal(m.passed_data[1]["x"], pd.Series([0., 1.], [2, 3]))

    def test_facet_categories_single_dim_shared(self):

        data = [
            ("a", 1, 1), ("b", 1, 1),
            ("a", 1, 2), ("c", 1, 2),
            ("b", 2, 1), ("d", 2, 1),
            ("e", 2, 2), ("e", 2, 1),
        ]
        df = pd.DataFrame(data, columns=["x", "row", "col"]).assign(y=1)
        variables = {k: k for k in df}

        m = MockMark()
        p = Plot(df, **variables).add(m).configure(sharex="row").plot()

        axs = p._figure.axes
        for ax in axs:
            assert ax.get_xticks() == [0, 1, 2]

        assert_vector_equal(m.passed_data[0]["x"], pd.Series([0., 1.], [0, 1]))
        assert_vector_equal(m.passed_data[1]["x"], pd.Series([0., 2.], [2, 3]))
        assert_vector_equal(m.passed_data[2]["x"], pd.Series([0., 1., 2.], [4, 5, 7]))
        assert_vector_equal(m.passed_data[3]["x"], pd.Series([2.], [6]))

    def test_pair_categories(self):

        data = [("a", "a"), ("b", "c")]
        df = pd.DataFrame(data, columns=["x1", "x2"]).assign(y=1)
        m = MockMark()
        p = Plot(df, y="y").pair(x=["x1", "x2"]).add(m).plot()

        ax1, ax2 = p._figure.axes
        assert ax1.get_xticks() == [0, 1]
        assert ax2.get_xticks() == [0, 1]
        assert_vector_equal(m.passed_data[0]["x"], pd.Series([0., 1.], [0, 1]))
        assert_vector_equal(m.passed_data[1]["x"], pd.Series([0., 1.], [0, 1]))

    @pytest.mark.xfail(
        LooseVersion(mpl.__version__) < "3.4.0",
        reason="Sharing paired categorical axes requires matplotlib>3.4.0"
    )
    def test_pair_categories_shared(self):

        data = [("a", "a"), ("b", "c")]
        df = pd.DataFrame(data, columns=["x1", "x2"]).assign(y=1)
        m = MockMark()
        p = Plot(df, y="y").pair(x=["x1", "x2"]).add(m).configure(sharex=True).plot()

        for ax in p._figure.axes:
            assert ax.get_xticks() == [0, 1, 2]
        assert_vector_equal(m.passed_data[0]["x"], pd.Series([0., 1.], [0, 1]))
        assert_vector_equal(m.passed_data[1]["x"], pd.Series([0., 2.], [0, 1]))

    def test_identity_mapping_linewidth(self):

        m = MockMark()
        x = y = [1, 2, 3, 4, 5]
        lw = pd.Series([.5, .1, .1, .9, 3])
        Plot(x=x, y=y, linewidth=lw).scale(linewidth=None).add(m).plot()
        for scales in m.passed_scales:
            assert_vector_equal(scales["linewidth"](lw), lw)

    # TODO where should RGB consistency be enforced?
    @pytest.mark.xfail(
        reason="Correct output representation for color with identity scale undefined"
    )
    def test_identity_mapping_color_strings(self):

        m = MockMark()
        x = y = [1, 2, 3]
        c = ["C0", "C2", "C1"]
        Plot(x=x, y=y, color=c).scale(color=None).add(m).plot()
        expected = mpl.colors.to_rgba_array(c)[:, :3]
        for scale in m.passed_scales:
            assert_array_equal(scale["color"](c), expected)

    def test_identity_mapping_color_tuples(self):

        m = MockMark()
        x = y = [1, 2, 3]
        c = [(1, 0, 0), (0, 1, 0), (1, 0, 0)]
        Plot(x=x, y=y, color=c).scale(color=None).add(m).plot()
        expected = mpl.colors.to_rgba_array(c)[:, :3]
        for scale in m.passed_scales:
            assert_array_equal(scale["color"](c), expected)

    @pytest.mark.xfail(
        reason="Need decision on what to do with scale defined for unused variable"
    )
    def test_undefined_variable_raises(self):

        p = Plot(x=[1, 2, 3], color=["a", "b", "c"]).scale(y=Continuous())
        err = r"No data found for variable\(s\) with explicit scale: {'y'}"
        with pytest.raises(RuntimeError, match=err):
            p.plot()


class TestPlotting:

    def test_matplotlib_object_creation(self):

        p = Plot().plot()
        assert isinstance(p._figure, mpl.figure.Figure)
        for sub in p._subplots:
            assert isinstance(sub["ax"], mpl.axes.Axes)

    def test_empty(self):

        m = MockMark()
        Plot().plot()
        assert m.n_splits == 0

    def test_single_split_single_layer(self, long_df):

        m = MockMark()
        p = Plot(long_df, x="f", y="z").add(m).plot()
        assert m.n_splits == 1

        assert m.passed_keys[0] == {}
        assert m.passed_axes == [sub["ax"] for sub in p._subplots]
        for col in p._data.frame:
            assert_series_equal(m.passed_data[0][col], p._data.frame[col])

    def test_single_split_multi_layer(self, long_df):

        vs = [{"color": "a", "width": "z"}, {"color": "b", "pattern": "c"}]

        class NoGroupingMark(MockMark):
            grouping_vars = []

        ms = [NoGroupingMark(), NoGroupingMark()]
        Plot(long_df).add(ms[0], **vs[0]).add(ms[1], **vs[1]).plot()

        for m, v in zip(ms, vs):
            for var, col in v.items():
                assert_vector_equal(m.passed_data[0][var], long_df[col])

    def check_splits_single_var(self, plot, mark, split_var, split_keys):

        assert mark.n_splits == len(split_keys)
        assert mark.passed_keys == [{split_var: key} for key in split_keys]

        full_data = plot._data.frame
        for i, key in enumerate(split_keys):

            split_data = full_data[full_data[split_var] == key]
            for col in split_data:
                assert_series_equal(mark.passed_data[i][col], split_data[col])

    def check_splits_multi_vars(self, plot, mark, split_vars, split_keys):

        assert mark.n_splits == np.prod([len(ks) for ks in split_keys])

        expected_keys = [
            dict(zip(split_vars, level_keys))
            for level_keys in itertools.product(*split_keys)
        ]
        assert mark.passed_keys == expected_keys

        full_data = plot._data.frame
        for i, keys in enumerate(itertools.product(*split_keys)):

            use_rows = pd.Series(True, full_data.index)
            for var, key in zip(split_vars, keys):
                use_rows &= full_data[var] == key
            split_data = full_data[use_rows]
            for col in split_data:
                assert_series_equal(mark.passed_data[i][col], split_data[col])

    @pytest.mark.parametrize(
        "split_var", [
            "color",  # explicitly declared on the Mark
            "group",  # implicitly used for all Mark classes
        ])
    def test_one_grouping_variable(self, long_df, split_var):

        split_col = "a"

        m = MockMark()
        p = Plot(long_df, x="f", y="z", **{split_var: split_col}).add(m).plot()

        split_keys = categorical_order(long_df[split_col])
        sub, *_ = p._subplots
        assert m.passed_axes == [sub["ax"] for _ in split_keys]
        self.check_splits_single_var(p, m, split_var, split_keys)

    def test_two_grouping_variables(self, long_df):

        split_vars = ["color", "group"]
        split_cols = ["a", "b"]
        variables = {var: col for var, col in zip(split_vars, split_cols)}

        m = MockMark()
        p = Plot(long_df, y="z", **variables).add(m).plot()

        split_keys = [categorical_order(long_df[col]) for col in split_cols]
        sub, *_ = p._subplots
        assert m.passed_axes == [
            sub["ax"] for _ in itertools.product(*split_keys)
        ]
        self.check_splits_multi_vars(p, m, split_vars, split_keys)

    def test_facets_no_subgroups(self, long_df):

        split_var = "col"
        split_col = "b"

        m = MockMark()
        p = Plot(long_df, x="f", y="z", **{split_var: split_col}).add(m).plot()

        split_keys = categorical_order(long_df[split_col])
        assert m.passed_axes == list(p._figure.axes)
        self.check_splits_single_var(p, m, split_var, split_keys)

    def test_facets_one_subgroup(self, long_df):

        facet_var, facet_col = "col", "a"
        group_var, group_col = "group", "b"

        m = MockMark()
        p = (
            Plot(long_df, x="f", y="z", **{group_var: group_col, facet_var: facet_col})
            .add(m)
            .plot()
        )

        split_keys = [categorical_order(long_df[col]) for col in [facet_col, group_col]]
        assert m.passed_axes == [
            ax
            for ax in list(p._figure.axes)
            for _ in categorical_order(long_df[group_col])
        ]
        self.check_splits_multi_vars(p, m, [facet_var, group_var], split_keys)

    def test_layer_specific_facet_disabling(self, long_df):

        axis_vars = {"x": "y", "y": "z"}
        row_var = "a"

        m = MockMark()
        p = Plot(long_df, **axis_vars, row=row_var).add(m, row=None).plot()

        col_levels = categorical_order(long_df[row_var])
        assert len(p._figure.axes) == len(col_levels)

        for data in m.passed_data:
            for var, col in axis_vars.items():
                assert_vector_equal(data[var], long_df[col])

    def test_paired_variables(self, long_df):

        x = ["x", "y"]
        y = ["f", "z"]

        m = MockMark()
        Plot(long_df).pair(x, y).add(m).plot()

        var_product = itertools.product(x, y)

        for data, (x_i, y_i) in zip(m.passed_data, var_product):
            assert_vector_equal(data["x"], long_df[x_i])
            assert_vector_equal(data["y"], long_df[y_i])

    def test_paired_one_dimension(self, long_df):

        x = ["y", "z"]

        m = MockMark()
        Plot(long_df).pair(x).add(m).plot()

        for data, x_i in zip(m.passed_data, x):
            assert_vector_equal(data["x"], long_df[x_i].astype(float))

    def test_paired_variables_one_subset(self, long_df):

        x = ["x", "y"]
        y = ["f", "z"]
        group = "a"

        long_df["x"] = long_df["x"].astype(float)  # simplify vector comparison

        m = MockMark()
        Plot(long_df, group=group).pair(x, y).add(m).plot()

        groups = categorical_order(long_df[group])
        var_product = itertools.product(x, y, groups)

        for data, (x_i, y_i, g_i) in zip(m.passed_data, var_product):
            rows = long_df[group] == g_i
            assert_vector_equal(data["x"], long_df.loc[rows, x_i])
            assert_vector_equal(data["y"], long_df.loc[rows, y_i])

    def test_paired_and_faceted(self, long_df):

        x = ["y", "z"]
        y = "f"
        row = "c"

        m = MockMark()
        Plot(long_df, y=y, row=row).pair(x).add(m).plot()

        facets = categorical_order(long_df[row])
        var_product = itertools.product(x, facets)

        for data, (x_i, f_i) in zip(m.passed_data, var_product):
            rows = long_df[row] == f_i
            assert_vector_equal(data["x"], long_df.loc[rows, x_i])
            assert_vector_equal(data["y"], long_df.loc[rows, y])

    def test_movement(self, long_df):

        orig_df = long_df.copy(deep=True)

        class MockMove(Move):
            def __call__(self, data, groupby, orient):
                return data.assign(x=data["x"] + 1)

        m = MockMark()
        Plot(long_df, x="z", y="z").add(m, move=MockMove()).plot()
        assert_vector_equal(m.passed_data[0]["x"], long_df["z"] + 1)
        assert_vector_equal(m.passed_data[0]["y"], long_df["z"])

        assert_frame_equal(long_df, orig_df)   # Test data was not mutated

    def test_movement_log_scale(self, long_df):

        class MockMove(Move):
            def __call__(self, data, groupby, orient):
                return data.assign(x=data["x"] - 1)

        m = MockMark()
        Plot(
            long_df, x="z", y="z"
        ).scale(x="log").add(m, move=MockMove()).plot()
        assert_vector_equal(m.passed_data[0]["x"], long_df["z"] / 10)

    def test_methods_clone(self, long_df):

        p1 = Plot(long_df, "x", "y")
        p2 = p1.add(MockMark()).facet("a")

        assert p1 is not p2
        assert not p1._layers
        assert not p1._facetspec

    def test_inplace(self, long_df):

        p1 = Plot(long_df, "x", "y")
        p2 = p1.inplace().add(MockMark())
        assert p2 is p1

        p3 = p2.inplace().add(MockMark())
        assert p3 is not p2

        p4 = p3.inplace(False).add(MockMark())
        assert p4 is not p3

        p5 = p4.inplace(True).add(MockMark())
        assert p5 is p4

    def test_default_is_no_pyplot(self):

        p = Plot().plot()

        assert not plt.get_fignums()
        assert isinstance(p._figure, mpl.figure.Figure)

    def test_with_pyplot(self):

        p = Plot().plot(pyplot=True)

        assert len(plt.get_fignums()) == 1
        fig = plt.gcf()
        assert p._figure is fig

    def test_show(self):

        p = Plot()

        with warnings.catch_warnings(record=True) as msg:
            out = p.show(block=False)
        assert out is None
        assert not hasattr(p, "_figure")

        assert len(plt.get_fignums()) == 1
        fig = plt.gcf()

        gui_backend = (
            # From https://github.com/matplotlib/matplotlib/issues/20281
            fig.canvas.manager.show != mpl.backend_bases.FigureManagerBase.show
        )
        if not gui_backend:
            assert msg

    def test_png_representation(self):

        p = Plot()
        data, metadata = p._repr_png_()

        assert not hasattr(p, "_figure")
        assert isinstance(data, bytes)
        assert imghdr.what("", data) == "png"
        assert sorted(metadata) == ["height", "width"]
        # TODO test retina scaling

    @pytest.mark.xfail(reason="Plot.save not yet implemented")
    def test_save(self):

        Plot().save()

    def test_on_axes(self):

        ax = mpl.figure.Figure().subplots()
        m = MockMark()
        p = Plot().on(ax).add(m).plot()
        assert m.passed_axes == [ax]
        assert p._figure is ax.figure

    @pytest.mark.parametrize("facet", [True, False])
    def test_on_figure(self, facet):

        f = mpl.figure.Figure()
        m = MockMark()
        p = Plot().on(f).add(m)
        if facet:
            p = p.facet(["a", "b"])
        p = p.plot()
        assert m.passed_axes == f.axes
        assert p._figure is f

    @pytest.mark.skipif(
        LooseVersion(mpl.__version__) < "3.4", reason="mpl<3.4 does not have SubFigure",
    )
    @pytest.mark.parametrize("facet", [True, False])
    def test_on_subfigure(self, facet):

        sf1, sf2 = mpl.figure.Figure().subfigures(2)
        sf1.subplots()
        m = MockMark()
        p = Plot().on(sf2).add(m)
        if facet:
            p = p.facet(["a", "b"])
        p = p.plot()
        assert m.passed_axes == sf2.figure.axes[1:]
        assert p._figure is sf2.figure

    def test_on_type_check(self):

        p = Plot()
        with pytest.raises(TypeError, match="The `Plot.on`.+<class 'list'>"):
            p.on([])

    def test_on_axes_with_subplots_error(self):

        ax = mpl.figure.Figure().subplots()

        p1 = Plot().facet(["a", "b"]).on(ax)
        with pytest.raises(RuntimeError, match="Cannot create multiple subplots"):
            p1.plot()

        p2 = Plot().pair([["a", "b"], ["x", "y"]]).on(ax)
        with pytest.raises(RuntimeError, match="Cannot create multiple subplots"):
            p2.plot()

    def test_axis_labels_from_constructor(self, long_df):

        ax, = Plot(long_df, x="a", y="b").plot()._figure.axes
        assert ax.get_xlabel() == "a"
        assert ax.get_ylabel() == "b"

        ax, = Plot(x=long_df["a"], y=long_df["b"].to_numpy()).plot()._figure.axes
        assert ax.get_xlabel() == "a"
        assert ax.get_ylabel() == ""

    def test_axis_labels_from_layer(self, long_df):

        m = MockMark()

        ax, = Plot(long_df).add(m, x="a", y="b").plot()._figure.axes
        assert ax.get_xlabel() == "a"
        assert ax.get_ylabel() == "b"

        p = Plot().add(m, x=long_df["a"], y=long_df["b"].to_list())
        ax, = p.plot()._figure.axes
        assert ax.get_xlabel() == "a"
        assert ax.get_ylabel() == ""

    def test_axis_labels_are_first_name(self, long_df):

        m = MockMark()
        p = (
            Plot(long_df, x=long_df["z"].to_list(), y="b")
            .add(m, x="a")
            .add(m, x="x", y="y")
        )
        ax, = p.plot()._figure.axes
        assert ax.get_xlabel() == "a"
        assert ax.get_ylabel() == "b"


class TestFacetInterface:

    @pytest.fixture(scope="class", params=["row", "col"])
    def dim(self, request):
        return request.param

    @pytest.fixture(scope="class", params=["reverse", "subset", "expand"])
    def reorder(self, request):
        return {
            "reverse": lambda x: x[::-1],
            "subset": lambda x: x[:-1],
            "expand": lambda x: x + ["z"],
        }[request.param]

    def check_facet_results_1d(self, p, df, dim, key, order=None):

        p = p.plot()

        order = categorical_order(df[key], order)
        assert len(p._figure.axes) == len(order)

        other_dim = {"row": "col", "col": "row"}[dim]

        for subplot, level in zip(p._subplots, order):
            assert subplot[dim] == level
            assert subplot[other_dim] is None
            assert subplot["ax"].get_title() == f"{key} = {level}"
            assert_gridspec_shape(subplot["ax"], **{f"n{dim}s": len(order)})

    def test_1d_from_init(self, long_df, dim):

        key = "a"
        p = Plot(long_df, **{dim: key})
        self.check_facet_results_1d(p, long_df, dim, key)

    def test_1d_from_facet(self, long_df, dim):

        key = "a"
        p = Plot(long_df).facet(**{dim: key})
        self.check_facet_results_1d(p, long_df, dim, key)

    def test_1d_from_init_as_vector(self, long_df, dim):

        key = "a"
        p = Plot(long_df, **{dim: long_df[key]})
        self.check_facet_results_1d(p, long_df, dim, key)

    def test_1d_from_facet_as_vector(self, long_df, dim):

        key = "a"
        p = Plot(long_df).facet(**{dim: long_df[key]})
        self.check_facet_results_1d(p, long_df, dim, key)

    def test_1d_from_init_with_order(self, long_df, dim, reorder):

        key = "a"
        order = reorder(categorical_order(long_df[key]))
        p = Plot(long_df, **{dim: key}).facet(order={dim: order})
        self.check_facet_results_1d(p, long_df, dim, key, order)

    def test_1d_from_facet_with_order(self, long_df, dim, reorder):

        key = "a"
        order = reorder(categorical_order(long_df[key]))
        p = Plot(long_df).facet(**{dim: key, "order": order})
        self.check_facet_results_1d(p, long_df, dim, key, order)

    def check_facet_results_2d(self, p, df, variables, order=None):

        p = p.plot()

        if order is None:
            order = {dim: categorical_order(df[key]) for dim, key in variables.items()}

        levels = itertools.product(*[order[dim] for dim in ["row", "col"]])
        assert len(p._subplots) == len(list(levels))

        for subplot, (row_level, col_level) in zip(p._subplots, levels):
            assert subplot["row"] == row_level
            assert subplot["col"] == col_level
            assert subplot["axes"].get_title() == (
                f"{variables['row']} = {row_level} | {variables['col']} = {col_level}"
            )
            assert_gridspec_shape(
                subplot["axes"], len(levels["row"]), len(levels["col"])
            )

    def test_2d_from_init(self, long_df):

        variables = {"row": "a", "col": "c"}
        p = Plot(long_df, **variables)
        self.check_facet_results_2d(p, long_df, variables)

    def test_2d_from_facet(self, long_df):

        variables = {"row": "a", "col": "c"}
        p = Plot(long_df).facet(**variables)
        self.check_facet_results_2d(p, long_df, variables)

    def test_2d_from_init_and_facet(self, long_df):

        variables = {"row": "a", "col": "c"}
        p = Plot(long_df, row=variables["row"]).facet(col=variables["col"])
        self.check_facet_results_2d(p, long_df, variables)

    def test_2d_from_facet_with_order(self, long_df, reorder):

        variables = {"row": "a", "col": "c"}
        order = {
            dim: reorder(categorical_order(long_df[key]))
            for dim, key in variables.items()
        }

        p = Plot(long_df).facet(**variables, order=order)
        self.check_facet_results_2d(p, long_df, variables, order)

    def test_axis_sharing(self, long_df):

        variables = {"row": "a", "col": "c"}

        p = Plot(long_df).facet(**variables)

        p1 = p.plot()
        root, *other = p1._figure.axes
        for axis in "xy":
            shareset = getattr(root, f"get_shared_{axis}_axes")()
            assert all(shareset.joined(root, ax) for ax in other)

        p2 = p.configure(sharex=False, sharey=False).plot()
        root, *other = p2._figure.axes
        for axis in "xy":
            shareset = getattr(root, f"get_shared_{axis}_axes")()
            assert not any(shareset.joined(root, ax) for ax in other)

        p3 = p.configure(sharex="col", sharey="row").plot()
        shape = (
            len(categorical_order(long_df[variables["row"]])),
            len(categorical_order(long_df[variables["col"]])),
        )
        axes_matrix = np.reshape(p3._figure.axes, shape)

        for (shared, unshared), vectors in zip(
            ["yx", "xy"], [axes_matrix, axes_matrix.T]
        ):
            for root, *other in vectors:
                shareset = {
                    axis: getattr(root, f"get_shared_{axis}_axes")() for axis in "xy"
                }
                assert all(shareset[shared].joined(root, ax) for ax in other)
                assert not any(shareset[unshared].joined(root, ax) for ax in other)

    def test_col_wrapping(self):

        cols = list("abcd")
        wrap = 3
        p = Plot().facet(col=cols, wrap=wrap).plot()

        assert len(p._figure.axes) == 4
        assert_gridspec_shape(p._figure.axes[0], len(cols) // wrap + 1, wrap)

        # TODO test axis labels and titles

    def test_row_wrapping(self):

        rows = list("abcd")
        wrap = 3
        p = Plot().facet(row=rows, wrap=wrap).plot()

        assert_gridspec_shape(p._figure.axes[0], wrap, len(rows) // wrap + 1)
        assert len(p._figure.axes) == 4

        # TODO test axis labels and titles


class TestPairInterface:

    def check_pair_grid(self, p, x, y):

        xys = itertools.product(y, x)

        for (y_i, x_j), subplot in zip(xys, p._subplots):

            ax = subplot["ax"]
            assert ax.get_xlabel() == "" if x_j is None else x_j
            assert ax.get_ylabel() == "" if y_i is None else y_i
            assert_gridspec_shape(subplot["ax"], len(y), len(x))

    @pytest.mark.parametrize(
        "vector_type", [list, np.array, pd.Series, pd.Index]
    )
    def test_all_numeric(self, long_df, vector_type):

        x, y = ["x", "y", "z"], ["s", "f"]
        p = Plot(long_df).pair(vector_type(x), vector_type(y)).plot()
        self.check_pair_grid(p, x, y)

    def test_single_variable_key_raises(self, long_df):

        p = Plot(long_df)
        err = "You must pass a sequence of variable keys to `y`"
        with pytest.raises(TypeError, match=err):
            p.pair(x=["x", "y"], y="z")

    @pytest.mark.parametrize("dim", ["x", "y"])
    def test_single_dimension(self, long_df, dim):

        variables = {"x": None, "y": None}
        variables[dim] = ["x", "y", "z"]
        p = Plot(long_df).pair(**variables).plot()
        variables = {k: [v] if v is None else v for k, v in variables.items()}
        self.check_pair_grid(p, **variables)

    def test_non_cartesian(self, long_df):

        x = ["x", "y"]
        y = ["f", "z"]

        p = Plot(long_df).pair(x, y, cartesian=False).plot()

        for i, subplot in enumerate(p._subplots):
            ax = subplot["ax"]
            assert ax.get_xlabel() == x[i]
            assert ax.get_ylabel() == y[i]
            assert_gridspec_shape(ax, 1, len(x))

        root, *other = p._figure.axes
        for axis in "xy":
            shareset = getattr(root, f"get_shared_{axis}_axes")()
            assert not any(shareset.joined(root, ax) for ax in other)

    def test_with_no_variables(self, long_df):

        all_cols = long_df.columns

        p1 = Plot(long_df).pair()
        for axis in "xy":
            assert p1._pairspec[axis] == all_cols.to_list()

        p2 = Plot(long_df, y="y").pair()
        assert all_cols.difference(p2._pairspec["x"]).item() == "y"
        assert "y" not in p2._pairspec

        p3 = Plot(long_df, color="a").pair()
        for axis in "xy":
            assert all_cols.difference(p3._pairspec[axis]).item() == "a"

        with pytest.raises(RuntimeError, match="You must pass `data`"):
            Plot().pair()

    def test_with_facets(self, long_df):

        x = "x"
        y = ["y", "z"]
        col = "a"

        p = Plot(long_df, x=x).facet(col).pair(y=y).plot()

        facet_levels = categorical_order(long_df[col])
        dims = itertools.product(y, facet_levels)

        for (y_i, col_i), subplot in zip(dims, p._subplots):

            ax = subplot["ax"]
            assert ax.get_xlabel() == x
            assert ax.get_ylabel() == y_i
            assert ax.get_title() == f"{col} = {col_i}"
            assert_gridspec_shape(ax, len(y), len(facet_levels))

    @pytest.mark.parametrize("variables", [("rows", "y"), ("columns", "x")])
    def test_error_on_facet_overlap(self, long_df, variables):

        facet_dim, pair_axis = variables
        p = Plot(long_df, **{facet_dim[:3]: "a"}).pair(**{pair_axis: ["x", "y"]})
        expected = f"Cannot facet the {facet_dim} while pairing on `{pair_axis}`."
        with pytest.raises(RuntimeError, match=expected):
            p.plot()

    @pytest.mark.parametrize("variables", [("columns", "y"), ("rows", "x")])
    def test_error_on_wrap_overlap(self, long_df, variables):

        facet_dim, pair_axis = variables
        p = (
            Plot(long_df, **{facet_dim[:3]: "a"})
            .facet(wrap=2)
            .pair(**{pair_axis: ["x", "y"]})
        )
        expected = f"Cannot wrap the {facet_dim} while pairing on `{pair_axis}``."
        with pytest.raises(RuntimeError, match=expected):
            p.plot()

    def test_axis_sharing(self, long_df):

        p = Plot(long_df).pair(x=["a", "b"], y=["y", "z"])
        shape = 2, 2

        p1 = p.plot()
        axes_matrix = np.reshape(p1._figure.axes, shape)

        for root, *other in axes_matrix:  # Test row-wise sharing
            x_shareset = getattr(root, "get_shared_x_axes")()
            assert not any(x_shareset.joined(root, ax) for ax in other)
            y_shareset = getattr(root, "get_shared_y_axes")()
            assert all(y_shareset.joined(root, ax) for ax in other)

        for root, *other in axes_matrix.T:  # Test col-wise sharing
            x_shareset = getattr(root, "get_shared_x_axes")()
            assert all(x_shareset.joined(root, ax) for ax in other)
            y_shareset = getattr(root, "get_shared_y_axes")()
            assert not any(y_shareset.joined(root, ax) for ax in other)

        p2 = p.configure(sharex=False, sharey=False).plot()
        root, *other = p2._figure.axes
        for axis in "xy":
            shareset = getattr(root, f"get_shared_{axis}_axes")()
            assert not any(shareset.joined(root, ax) for ax in other)

    def test_axis_sharing_with_facets(self, long_df):

        p = Plot(long_df, y="y").pair(x=["a", "b"]).facet(row="c").plot()
        shape = 2, 2

        axes_matrix = np.reshape(p._figure.axes, shape)

        for root, *other in axes_matrix:  # Test row-wise sharing
            x_shareset = getattr(root, "get_shared_x_axes")()
            assert not any(x_shareset.joined(root, ax) for ax in other)
            y_shareset = getattr(root, "get_shared_y_axes")()
            assert all(y_shareset.joined(root, ax) for ax in other)

        for root, *other in axes_matrix.T:  # Test col-wise sharing
            x_shareset = getattr(root, "get_shared_x_axes")()
            assert all(x_shareset.joined(root, ax) for ax in other)
            y_shareset = getattr(root, "get_shared_y_axes")()
            assert all(y_shareset.joined(root, ax) for ax in other)

    def test_x_wrapping(self, long_df):

        x_vars = ["f", "x", "y", "z"]
        wrap = 3
        p = Plot(long_df, y="y").pair(x=x_vars, wrap=wrap).plot()

        assert_gridspec_shape(p._figure.axes[0], len(x_vars) // wrap + 1, wrap)
        assert len(p._figure.axes) == len(x_vars)

        # TODO test axis labels and visibility

    def test_y_wrapping(self, long_df):

        y_vars = ["f", "x", "y", "z"]
        wrap = 3
        p = Plot(long_df, x="x").pair(y=y_vars, wrap=wrap).plot()

        assert_gridspec_shape(p._figure.axes[0], wrap, len(y_vars) // wrap + 1)
        assert len(p._figure.axes) == len(y_vars)

        # TODO test axis labels and visibility

    def test_noncartesian_wrapping(self, long_df):

        x_vars = ["a", "b", "c", "t"]
        y_vars = ["f", "x", "y", "z"]
        wrap = 3

        p = (
            Plot(long_df, x="x")
            .pair(x=x_vars, y=y_vars, wrap=wrap, cartesian=False)
            .plot()
        )

        assert_gridspec_shape(p._figure.axes[0], len(x_vars) // wrap + 1, wrap)
        assert len(p._figure.axes) == len(x_vars)

    def test_orient_inference(self, long_df):

        orient_list = []

        class CaptureMoveOrient(Move):
            def __call__(self, data, groupby, orient):
                orient_list.append(orient)
                return data

        (
            Plot(long_df, x="x")
            .pair(y=["b", "z"])
            .add(MockMark(), move=CaptureMoveOrient())
            .plot()
        )

        assert orient_list == ["y", "x"]


class TestLabelVisibility:

    def test_single_subplot(self, long_df):

        x, y = "a", "z"
        p = Plot(long_df, x=x, y=y).plot()
        subplot, *_ = p._subplots
        ax = subplot["ax"]
        assert ax.xaxis.get_label().get_visible()
        assert ax.yaxis.get_label().get_visible()
        assert all(t.get_visible() for t in ax.get_xticklabels())
        assert all(t.get_visible() for t in ax.get_yticklabels())

    @pytest.mark.parametrize(
        "facet_kws,pair_kws", [({"col": "b"}, {}), ({}, {"x": ["x", "y", "f"]})]
    )
    def test_1d_column(self, long_df, facet_kws, pair_kws):

        x = None if "x" in pair_kws else "a"
        y = "z"
        p = Plot(long_df, x=x, y=y).plot()
        first, *other = p._subplots

        ax = first["ax"]
        assert ax.xaxis.get_label().get_visible()
        assert ax.yaxis.get_label().get_visible()
        assert all(t.get_visible() for t in ax.get_xticklabels())
        assert all(t.get_visible() for t in ax.get_yticklabels())

        for s in other:
            ax = s["ax"]
            assert ax.xaxis.get_label().get_visible()
            assert not ax.yaxis.get_label().get_visible()
            assert all(t.get_visible() for t in ax.get_xticklabels())
            assert not any(t.get_visible() for t in ax.get_yticklabels())

    @pytest.mark.parametrize(
        "facet_kws,pair_kws", [({"row": "b"}, {}), ({}, {"y": ["x", "y", "f"]})]
    )
    def test_1d_row(self, long_df, facet_kws, pair_kws):

        x = "z"
        y = None if "y" in pair_kws else "z"
        p = Plot(long_df, x=x, y=y).plot()
        first, *other = p._subplots

        ax = first["ax"]
        assert ax.xaxis.get_label().get_visible()
        assert all(t.get_visible() for t in ax.get_xticklabels())
        assert ax.yaxis.get_label().get_visible()
        assert all(t.get_visible() for t in ax.get_yticklabels())

        for s in other:
            ax = s["ax"]
            assert not ax.xaxis.get_label().get_visible()
            assert ax.yaxis.get_label().get_visible()
            assert not any(t.get_visible() for t in ax.get_xticklabels())
            assert all(t.get_visible() for t in ax.get_yticklabels())

    def test_1d_column_wrapped(self):

        p = Plot().facet(col=["a", "b", "c", "d"], wrap=3).plot()
        subplots = list(p._subplots)

        for s in [subplots[0], subplots[-1]]:
            ax = s["ax"]
            assert ax.yaxis.get_label().get_visible()
            assert all(t.get_visible() for t in ax.get_yticklabels())

        for s in subplots[1:]:
            ax = s["ax"]
            assert ax.xaxis.get_label().get_visible()
            assert all(t.get_visible() for t in ax.get_xticklabels())

        for s in subplots[1:-1]:
            ax = s["ax"]
            assert not ax.yaxis.get_label().get_visible()
            assert not any(t.get_visible() for t in ax.get_yticklabels())

        ax = subplots[0]["ax"]
        assert not ax.xaxis.get_label().get_visible()
        assert not any(t.get_visible() for t in ax.get_xticklabels())

    def test_1d_row_wrapped(self):

        p = Plot().facet(row=["a", "b", "c", "d"], wrap=3).plot()
        subplots = list(p._subplots)

        for s in subplots[:-1]:
            ax = s["ax"]
            assert ax.yaxis.get_label().get_visible()
            assert all(t.get_visible() for t in ax.get_yticklabels())

        for s in subplots[-2:]:
            ax = s["ax"]
            assert ax.xaxis.get_label().get_visible()
            assert all(t.get_visible() for t in ax.get_xticklabels())

        for s in subplots[:-2]:
            ax = s["ax"]
            assert not ax.xaxis.get_label().get_visible()
            assert not any(t.get_visible() for t in ax.get_xticklabels())

        ax = subplots[-1]["ax"]
        assert not ax.yaxis.get_label().get_visible()
        assert not any(t.get_visible() for t in ax.get_yticklabels())

    def test_1d_column_wrapped_noncartesian(self, long_df):

        p = (
            Plot(long_df)
            .pair(x=["a", "b", "c"], y=["x", "y", "z"], wrap=2, cartesian=False)
            .plot()
        )
        for s in p._subplots:
            ax = s["ax"]
            assert ax.xaxis.get_label().get_visible()
            assert all(t.get_visible() for t in ax.get_xticklabels())
            assert ax.yaxis.get_label().get_visible()
            assert all(t.get_visible() for t in ax.get_yticklabels())

    def test_2d(self):

        p = Plot().facet(col=["a", "b"], row=["x", "y"]).plot()
        subplots = list(p._subplots)

        for s in subplots[:2]:
            ax = s["ax"]
            assert not ax.xaxis.get_label().get_visible()
            assert not any(t.get_visible() for t in ax.get_xticklabels())

        for s in subplots[2:]:
            ax = s["ax"]
            assert ax.xaxis.get_label().get_visible()
            assert all(t.get_visible() for t in ax.get_xticklabels())

        for s in [subplots[0], subplots[2]]:
            ax = s["ax"]
            assert ax.yaxis.get_label().get_visible()
            assert all(t.get_visible() for t in ax.get_yticklabels())

        for s in [subplots[1], subplots[3]]:
            ax = s["ax"]
            assert not ax.yaxis.get_label().get_visible()
            assert not any(t.get_visible() for t in ax.get_yticklabels())

    def test_2d_unshared(self):

        p = (
            Plot()
            .facet(col=["a", "b"], row=["x", "y"])
            .configure(sharex=False, sharey=False)
            .plot()
        )
        subplots = list(p._subplots)

        for s in subplots[:2]:
            ax = s["ax"]
            assert not ax.xaxis.get_label().get_visible()
            assert all(t.get_visible() for t in ax.get_xticklabels())

        for s in subplots[2:]:
            ax = s["ax"]
            assert ax.xaxis.get_label().get_visible()
            assert all(t.get_visible() for t in ax.get_xticklabels())

        for s in [subplots[0], subplots[2]]:
            ax = s["ax"]
            assert ax.yaxis.get_label().get_visible()
            assert all(t.get_visible() for t in ax.get_yticklabels())

        for s in [subplots[1], subplots[3]]:
            ax = s["ax"]
            assert not ax.yaxis.get_label().get_visible()
            assert all(t.get_visible() for t in ax.get_yticklabels())


class TestLegend:

    @pytest.fixture
    def xy(self):
        return dict(x=[1, 2, 3, 4], y=[1, 2, 3, 4])

    def test_single_layer_single_variable(self, xy):

        s = pd.Series(["a", "b", "a", "c"], name="s")
        p = Plot(**xy).add(MockMark(), color=s).plot()
        e, = p._legend_contents

        labels = categorical_order(s)

        assert e[0] == (s.name, s.name)
        assert e[-1] == labels

        artists = e[1]
        assert len(artists) == len(labels)
        for a, label in zip(artists, labels):
            assert isinstance(a, mpl.artist.Artist)
            assert a.value == label
            assert a.variables == ["color"]

    def test_single_layer_common_variable(self, xy):

        s = pd.Series(["a", "b", "a", "c"], name="s")
        sem = dict(color=s, marker=s)
        p = Plot(**xy).add(MockMark(), **sem).plot()
        e, = p._legend_contents

        labels = categorical_order(s)

        assert e[0] == (s.name, s.name)
        assert e[-1] == labels

        artists = e[1]
        assert len(artists) == len(labels)
        for a, label in zip(artists, labels):
            assert isinstance(a, mpl.artist.Artist)
            assert a.value == label
            assert a.variables == list(sem)

    def test_single_layer_common_unnamed_variable(self, xy):

        s = np.array(["a", "b", "a", "c"])
        sem = dict(color=s, marker=s)
        p = Plot(**xy).add(MockMark(), **sem).plot()

        e, = p._legend_contents

        labels = list(np.unique(s))  # assumes sorted order

        assert e[0] == (None, id(s))
        assert e[-1] == labels

        artists = e[1]
        assert len(artists) == len(labels)
        for a, label in zip(artists, labels):
            assert isinstance(a, mpl.artist.Artist)
            assert a.value == label
            assert a.variables == list(sem)

    def test_single_layer_multi_variable(self, xy):

        s1 = pd.Series(["a", "b", "a", "c"], name="s1")
        s2 = pd.Series(["m", "m", "p", "m"], name="s2")
        sem = dict(color=s1, marker=s2)
        p = Plot(**xy).add(MockMark(), **sem).plot()
        e1, e2 = p._legend_contents

        variables = {v.name: k for k, v in sem.items()}

        for e, s in zip([e1, e2], [s1, s2]):
            assert e[0] == (s.name, s.name)

            labels = categorical_order(s)
            assert e[-1] == labels

            artists = e[1]
            assert len(artists) == len(labels)
            for a, label in zip(artists, labels):
                assert isinstance(a, mpl.artist.Artist)
                assert a.value == label
                assert a.variables == [variables[s.name]]

    def test_multi_layer_single_variable(self, xy):

        s = pd.Series(["a", "b", "a", "c"], name="s")
        p = Plot(**xy, color=s).add(MockMark()).add(MockMark()).plot()
        e1, e2 = p._legend_contents

        labels = categorical_order(s)

        for e in [e1, e2]:
            assert e[0] == (s.name, s.name)

            labels = categorical_order(s)
            assert e[-1] == labels

            artists = e[1]
            assert len(artists) == len(labels)
            for a, label in zip(artists, labels):
                assert isinstance(a, mpl.artist.Artist)
                assert a.value == label
                assert a.variables == ["color"]

    def test_multi_layer_multi_variable(self, xy):

        s1 = pd.Series(["a", "b", "a", "c"], name="s1")
        s2 = pd.Series(["m", "m", "p", "m"], name="s2")
        sem = dict(color=s1), dict(marker=s2)
        variables = {"s1": "color", "s2": "marker"}
        p = Plot(**xy).add(MockMark(), **sem[0]).add(MockMark(), **sem[1]).plot()
        e1, e2 = p._legend_contents

        for e, s in zip([e1, e2], [s1, s2]):
            assert e[0] == (s.name, s.name)

            labels = categorical_order(s)
            assert e[-1] == labels

            artists = e[1]
            assert len(artists) == len(labels)
            for a, label in zip(artists, labels):
                assert isinstance(a, mpl.artist.Artist)
                assert a.value == label
                assert a.variables == [variables[s.name]]

    def test_identity_scale_ignored(self, xy):

        s = pd.Series(["r", "g", "b", "g"])
        p = Plot(**xy).add(MockMark(), color=s).scale(color=None).plot()
        assert not p._legend_contents

    # TODO test actually legend content? But wait until we decide
    # how we want to actually create the legend ...
