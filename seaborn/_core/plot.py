from __future__ import annotations

import io
import re
import itertools
from copy import deepcopy
from distutils.version import LooseVersion

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt  # TODO defer import into Plot.show()

from seaborn._compat import scale_factory, set_scale_obj
from seaborn._core.rules import categorical_order
from seaborn._core.data import PlotData
from seaborn._core.subplots import Subplots
from seaborn._core.mappings import (
    ColorSemantic,
    BooleanSemantic,
    MarkerSemantic,
    LineStyleSemantic,
    LineWidthSemantic,
    AlphaSemantic,
    PointSizeSemantic,
    WidthSemantic,
    IdentityMapping,
)
from seaborn._core.scales import (
    Scale,
    NumericScale,
    CategoricalScale,
    DateTimeScale,
    IdentityScale,
    get_default_scale,
)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Literal, Any
    from collections.abc import Callable, Generator, Iterable, Hashable
    from pandas import DataFrame, Series, Index
    from matplotlib.axes import Axes
    from matplotlib.artist import Artist
    from matplotlib.color import Normalize
    from matplotlib.figure import Figure, SubFigure
    from matplotlib.scale import ScaleBase
    from seaborn._core.mappings import Semantic, SemanticMapping
    from seaborn._marks.base import Mark
    from seaborn._stats.base import Stat
    from seaborn._core.typing import (
        DataSource,
        PaletteSpec,
        VariableSpec,
        OrderSpec,
        NormSpec,
        DiscreteValueSpec,
        ContinuousValueSpec,
    )


SEMANTICS = {  # TODO should this be pluggable?
    "color": ColorSemantic(),
    "fillcolor": ColorSemantic(variable="fillcolor"),
    "alpha": AlphaSemantic(),
    "fillalpha": AlphaSemantic(variable="fillalpha"),
    "edgecolor": ColorSemantic(variable="edgecolor"),
    "fill": BooleanSemantic(values=None, variable="fill"),
    "marker": MarkerSemantic(),
    "linestyle": LineStyleSemantic(),
    "linewidth": LineWidthSemantic(),
    "pointsize": PointSizeSemantic(),

    # TODO we use this dictionary to access the standardize_value method
    # in Mark.resolve, even though these are not really "semantics" as such
    # (or are they?); we might want to introduce a different concept?
    # Maybe call this VARIABLES and have e.g. ColorSemantic, BaselineVariable?
    "width": WidthSemantic(),
}


class Plot:

    _data: PlotData
    _layers: list[dict]
    _semantics: dict[str, Semantic]
    _scales: dict[str, Scale]

    # TODO use TypedDict here
    _subplotspec: dict[str, Any]
    _facetspec: dict[str, Any]
    _pairspec: dict[str, Any]

    def __init__(
        self,
        data: DataSource = None,
        **variables: VariableSpec,
    ):

        # TODO accept x, y as args?

        self._data = PlotData(data, variables)
        self._layers = []

        self._scales = {}
        self._semantics = {}

        self._subplotspec = {}
        self._facetspec = {}
        self._pairspec = {}

        self._target = None

    def _repr_png_(self) -> tuple[bytes, dict[str, float]]:

        return self.plot()._repr_png_()

    # TODO _repr_svg_?

    def on(self, target: Axes | SubFigure | Figure) -> Plot:

        accepted_types: tuple  # Allow tuple of various length
        if hasattr(mpl.figure, "SubFigure"):  # Added in mpl 3.4
            accepted_types = (
                mpl.axes.Axes, mpl.figure.SubFigure, mpl.figure.Figure
            )
            accepted_types_str = (
                f"{mpl.axes.Axes}, {mpl.figure.SubFigure}, or {mpl.figure.Figure}"
            )
        else:
            accepted_types = mpl.axes.Axes, mpl.figure.Figure
            accepted_types_str = f"{mpl.axes.Axes} or {mpl.figure.Figure}"

        if not isinstance(target, accepted_types):
            err = (
                f"The `Plot.on` target must be an instance of {accepted_types_str}. "
                f"You passed an object of class {target.__class__} instead."
            )
            raise TypeError(err)

        self._target = target

        return self

    def add(
        self,
        mark: Mark,
        stat: Stat | None = None,
        orient: Literal["x", "y", "v", "h"] | None = None,
        data: DataSource = None,
        **variables: VariableSpec,
    ) -> Plot:

        # TODO do a check here that mark has been initialized,
        # otherwise errors will be inscrutable

        # TODO currently it doesn't work to specify faceting for the first time in add()
        # and I think this would be too difficult. But it should not silently fail.

        if stat is None and mark.default_stat is not None:
            # TODO We need some way to say "do no stat transformation" that is different
            # from "use the default". That's basically an IdentityStat.
            # TODO when fixed see FIXME:IdentityStat

            # Default stat needs to be initialized here so that its state is
            # not modified across multiple plots. If a Mark wants to define a default
            # stat with non-default params, it should use functools.partial
            stat = mark.default_stat()

        self._layers.append({
            "mark": mark,
            "stat": stat,
            "source": data,
            "variables": variables,
            "orient": {"v": "x", "h": "y"}.get(orient, orient),  # type: ignore
        })

        return self

    def pair(
        self,
        x: list[Hashable] | Index[Hashable] | None = None,
        y: list[Hashable] | Index[Hashable] | None = None,
        wrap: int | None = None,
        cartesian: bool = True,  # TODO bikeshed name, maybe cross?
        # TODO other existing PairGrid things like corner?
    ) -> Plot:

        # TODO Problems to solve:
        #
        # - Unclear is how to handle the diagonal plots that PairGrid offers
        #
        # - Implementing this will require lots of downscale changes in figure setup,
        #   and especially the axis scaling, which will need to be pair specific

        # TODO lists of vectors currently work, but I'm not sure where best to test

        # TODO add data kwarg here? (it's everywhere else...)

        # TODO is is weird to call .pair() to create univariate plots?
        # i.e. Plot(data).pair(x=[...]). The basic logic is fine.
        # But maybe a different verb (e.g. Plot.spread) would be more clear?
        # Then Plot(data).pair(x=[...]) would show the given x vars vs all.

        pairspec: dict[str, Any] = {}

        if x is None and y is None:

            # Default to using all columns in the input source data, aside from
            # those that were assigned to a variable in the constructor
            # TODO Do we want to allow additional filtering by variable type?
            # (Possibly even default to using only numeric columns)

            if self._data._source_data is None:
                err = "You must pass `data` in the constructor to use default pairing."
                raise RuntimeError(err)

            all_unused_columns = [
                key for key in self._data._source_data
                if key not in self._data.names.values()
            ]
            for axis in "xy":
                if axis not in self._data:
                    pairspec[axis] = all_unused_columns
        else:

            axes = {"x": x, "y": y}
            for axis, arg in axes.items():
                if arg is not None:
                    if isinstance(arg, (str, int)):
                        err = f"You must pass a sequence of variable keys to `{axis}`"
                        raise TypeError(err)
                    pairspec[axis] = list(arg)

        pairspec["variables"] = {}
        pairspec["structure"] = {}
        for axis in "xy":
            keys = []
            for i, col in enumerate(pairspec.get(axis, [])):
                # TODO note that this assumes no variables are defined as {axis}{digit}
                # This could be a slight problem as matplotlib occasionally uses that
                # format for artists that take multiple parameters on each axis.
                # Perhaps we should set the internal pair variables to "_{axis}{index}"?
                key = f"{axis}{i}"
                keys.append(key)
                pairspec["variables"][key] = col

            if keys:
                pairspec["structure"][axis] = keys

        # TODO raise here if cartesian is False and len(x) != len(y)?
        pairspec["cartesian"] = cartesian
        pairspec["wrap"] = wrap

        self._pairspec.update(pairspec)
        return self

    def facet(
        self,
        # TODO require kwargs?
        col: VariableSpec = None,
        row: VariableSpec = None,
        order: OrderSpec | dict[Literal["col", "row"], OrderSpec] = None,
        wrap: int | None = None,
    ) -> Plot:

        # Can't pass `None` here or it will disinherit the `Plot()` def
        variables = {}
        if col is not None:
            variables["col"] = col
        if row is not None:
            variables["row"] = row

        col_order = row_order = None
        if isinstance(order, dict):
            col_order = order.get("col")
            if col_order is not None:
                col_order = list(col_order)
            row_order = order.get("row")
            if row_order is not None:
                row_order = list(row_order)
        elif order is not None:
            # TODO Allow order: list here when single facet var defined in constructor?
            if col is not None:
                col_order = list(order)
            if row is not None:
                row_order = list(order)

        self._facetspec.update({
            "source": None,
            "variables": variables,
            "col_order": col_order,
            "row_order": row_order,
            "wrap": wrap,
        })

        return self

    def map_color(
        self,
        # TODO accept variable specification here?
        palette: PaletteSpec = None,
        order: OrderSpec = None,
        norm: NormSpec = None,
    ) -> Plot:

        # TODO we do some fancy business currently to avoid having to
        # write these ... do we want that to persist or is it too confusing?
        # If we do ... maybe we don't even need to write these methods, but can
        # instead programatically add them based on central dict of mapping objects.
        # ALSO TODO should these be initialized with defaults?
        # TODO if we define default semantics, we can use that
        # for initialization and make this more abstract (assuming kwargs match?)
        self._semantics["color"] = ColorSemantic(palette)
        self._scale_from_map("color", palette, order)
        return self

    def map_alpha(
        self,
        values: ContinuousValueSpec = None,
        order: OrderSpec | None = None,
        norm: Normalize | None = None,
    ) -> Plot:

        self._semantics["alpha"] = AlphaSemantic(values, variable="alpha")
        self._scale_from_map("alpha", values, order, norm)
        return self

    def map_fillcolor(
        self,
        palette: PaletteSpec = None,
        order: OrderSpec = None,
        norm: NormSpec = None,
    ) -> Plot:

        self._semantics["fillcolor"] = ColorSemantic(palette, variable="fillcolor")
        self._scale_from_map("fillcolor", palette, order)
        return self

    def map_fillalpha(
        self,
        values: ContinuousValueSpec = None,
        order: OrderSpec | None = None,
        norm: Normalize | None = None,
    ) -> Plot:

        self._semantics["fillalpha"] = AlphaSemantic(values, variable="fillalpha")
        self._scale_from_map("fillalpha", values, order, norm)
        return self

    def map_fill(
        self,
        values: DiscreteValueSpec = None,
        order: OrderSpec = None,
    ) -> Plot:

        self._semantics["fill"] = BooleanSemantic(values, variable="fill")
        self._scale_from_map("fill", values, order)
        return self

    def map_marker(
        self,
        shapes: DiscreteValueSpec = None,
        order: OrderSpec = None,
    ) -> Plot:

        self._semantics["marker"] = MarkerSemantic(shapes, variable="marker")
        self._scale_from_map("linewidth", shapes, order)
        return self

    def map_linestyle(
        self,
        styles: DiscreteValueSpec = None,
        order: OrderSpec = None,
    ) -> Plot:

        self._semantics["linestyle"] = LineStyleSemantic(styles, variable="linestyle")
        self._scale_from_map("linewidth", styles, order)
        return self

    def map_linewidth(
        self,
        values: ContinuousValueSpec = None,
        order: OrderSpec | None = None,
        norm: Normalize | None = None,
        # TODO clip?
    ) -> Plot:

        self._semantics["linewidth"] = LineWidthSemantic(values, variable="linewidth")
        self._scale_from_map("linewidth", values, order, norm)
        return self

    def _scale_from_map(self, var, values, order, norm=None) -> None:

        if order is not None:
            self.scale_categorical(var, order=order)
        elif norm is not None:
            if isinstance(values, (dict, list)):
                values_type = type(values).__name__
                err = f"Cannot use a norm with a {values_type} of {var} values."
                raise ValueError(err)
            self.scale_numeric(var, norm=norm)

    # TODO have map_gradient?
    # This could be used to add another color-like dimension
    # and also the basis for what mappings like stat.density -> rgba do

    # TODO map_saturation/map_chroma as a binary semantic?

    # The scale function names are a bit verbose. Two other options are:
    # - Have shorthand names (scale_num / scale_cat / scale_dt / scale_id)
    # - Have a separate scale(var, scale, norm, order, formatter, ...) method
    #   that dispatches based on the arguments it gets; keep the verbose methods
    #   around for use in case of ambiguity (e.g. to force a numeric variable to
    #   get a categorical scale without defining an order for it.

    def scale_numeric(
        self,
        var: str,
        scale: str | ScaleBase = "linear",
        norm: NormSpec = None,
        # TODO add clip? Useful for e.g., making sure lines don't get too thick.
        # (If we add clip, should we make the legend say like ``> value`)?
        **kwargs  # Needed? Or expose what we need?
    ) -> Plot:

        # TODO use norm for setting axis limits? Or otherwise share an interface?
        # Or separate norm as a Normalize object and limits as a tuple?
        # (If we have one we can create the other)

        # TODO Do we want to be able to call this on numbers-as-strings data and
        # have it work sensibly?

        if scale == "log":
            # TODO document that passing a LogNorm without this set can cause issues
            # (It's not a public attribute on the scale/transform)
            kwargs.setdefault("nonpositive", "mask")

        if not isinstance(scale, mpl.scale.ScaleBase):
            scale = scale_factory(scale, var, **kwargs)
        self._scales[var] = NumericScale(scale, norm)

        return self

    def scale_categorical(  # TODO FIXME:names scale_cat()?
        self,
        var: str,
        order: Series | Index | Iterable | None = None,
        # TODO parameter for binning continuous variable?
        formatter: Callable[[Any], str] = format,
    ) -> Plot:

        # TODO format() is not a great default for formatter(), ideally we'd want a
        # function that produces a "minimal" representation for numeric data and dates.
        # e.g.
        # 0.3333333333 -> 0.33 (maybe .2g?) This is trickiest
        # 1.0 -> 1
        # 2000-01-01 01:01:000000 -> "2000-01-01", or even "Jan 2000" for monthly data

        # Note that this will need to be chosen at setup() time as I think we
        # want the minimal representation for *all* values, not each one
        # individually.  There is also a subtle point about the difference
        # between what shows up in the ticks when a coordinate variable is
        # categorical vs what shows up in a legend.

        # TODO how to set limits/margins "nicely"? (i.e. 0.5 data units, past extremes)
        # TODO similarly, should this modify grid state like current categorical plots?
        # TODO "smart"/data-dependant ordering (e.g. order by median of y variable)

        if order is not None:
            order = list(order)

        scale = mpl.scale.LinearScale(var)
        self._scales[var] = CategoricalScale(scale, order, formatter)
        return self

    def scale_datetime(
        self,
        var: str,
        norm: Normalize | tuple[Any, Any] | None = None,
    ) -> Plot:

        scale = mpl.scale.LinearScale(var)
        self._scales[var] = DateTimeScale(scale, norm)

        # TODO I think rather than dealing with the question of "should we follow
        # pandas or matplotlib conventions with float -> date conversion, we should
        # force the user to provide a unit when calling this with a numeric variable.

        # TODO what else should this do?
        # We should pass kwargs to the DateTime cast probably.
        # Should we also explicitly expose more of the pd.to_datetime interface?

        # TODO also we should be able to set the formatter here
        # (well, and also in the other scale methods)
        # But it's especially important here because the default matplotlib formatter
        # is not very nice, and we don't need to be bound by that, so we should probably
        # (1) use fewer minticks
        # (2) use the concise dateformatter by default

        return self

    def scale_identity(self, var: str) -> Plot:

        self._scales[var] = IdentityScale()
        return self

    def configure(
        self,
        figsize: tuple[float, float] | None = None,
        sharex: bool | Literal["row", "col"] | None = None,
        sharey: bool | Literal["row", "col"] | None = None,
    ) -> Plot:

        # TODO add an "auto" mode for figsize that roughly scales with the rcParams
        # figsize (so that works), but expands to prevent subplots from being squished
        # Also should we have height=, aspect=, exclusive with figsize? Or working
        # with figsize when only one is defined?

        # TODO figsize has no actual effect here
        self._figsize = figsize

        subplot_keys = ["sharex", "sharey"]
        for key in subplot_keys:
            val = locals()[key]
            if val is not None:
                self._subplotspec[key] = val

        return self

    # TODO def legend (ugh)

    def theme(self) -> Plot:

        # TODO Plot-specific themes using the seaborn theming system
        # TODO should this also be where custom figure size goes?
        raise NotImplementedError
        return self

    # TODO decorate? (or similar, for various texts) alt names: label?

    def clone(self) -> Plot:

        if self._target is not None:
            # TODO think about whether this restriction is needed with immutable Plot
            raise RuntimeError("Cannot clone after calling `Plot.on`.")
        # TODO we are moving towards non-mutatable Plot so we don't need deep copy here
        return deepcopy(self)

    def save(self, fname, **kwargs) -> Plot:
        # TODO kws?
        self.plot().save(fname, **kwargs)
        return self

    def plot(self, pyplot=False) -> Plotter:

        # TODO if we have _target object, pyplot should be determined by whether it
        # is hooked into the pyplot state machine (how do we check?)

        plotter = Plotter(pyplot=pyplot)
        plotter._setup_data(self)
        plotter._setup_figure(self)
        plotter._setup_scales(self)
        plotter._setup_mappings(self)

        for layer in plotter._layers:
            plotter._plot_layer(self, layer)

        # TODO should this go here?
        plotter._make_legend()  # TODO does this return?

        # TODO this should be configurable
        if not plotter._figure.get_constrained_layout():
            plotter._figure.set_tight_layout(True)

        return plotter

    def show(self, **kwargs) -> None:

        # TODO make pyplot configurable at the class level, and when not using,
        # import IPython.display and call on self to populate cell output?

        # Keep an eye on whether matplotlib implements "attaching" an existing
        # figure to pyplot: https://github.com/matplotlib/matplotlib/pull/14024
        if self._target is None:
            self.clone().plot(pyplot=True)
        else:
            self.plot(pyplot=True)
        plt.show(**kwargs)


class Plotter:

    _mappings: dict[str, SemanticMapping]

    def __init__(self, pyplot=False):

        self.pyplot = pyplot
        self._legend_contents: list[
            tuple[str, str | int], list[Artist], list[str],
        ] = []

    def save(self, fname, **kwargs) -> Plotter:
        # TODO type fname as string or path; handle Path objects if matplotlib can't
        kwargs.setdefault("dpi", 96)
        self._figure.savefig(fname, **kwargs)
        return self

    def show(self, **kwargs) -> None:
        # TODO if we did not create the Plotter with pyplot, is it possible to do this?
        # If not we should clearly raise.
        plt.show(**kwargs)

    # TODO API for accessing the underlying matplotlib objects
    # TODO what else is useful in the public API for this class?

    # def draw?

    def _repr_png_(self) -> tuple[bytes, dict[str, float]]:

        # TODO better to do this through a Jupyter hook? e.g.
        # ipy = IPython.core.formatters.get_ipython()
        # fmt = ipy.display_formatter.formatters["text/html"]
        # fmt.for_type(Plot, ...)
        # Would like to have a svg option too, not sure how to make that flexible

        # TODO use matplotlib backend directly instead of going through savefig?

        # TODO perhaps have self.show() flip a switch to disable this, so that
        # user does not end up with two versions of the figure in the output

        # TODO use bbox_inches="tight" like the inline backend?
        # pro: better results,  con: (sometimes) confusing results
        # Better solution would be to default (with option to change)
        # to using constrained/tight layout.

        # TODO need to decide what the right default behavior here is:
        # - Use dpi=72 to match default InlineBackend figure size?
        # - Accept a generic "scaling" somewhere and scale DPI from that,
        #   either with 1x -> 72 or 1x -> 96 and the default scaling be .75?
        # - Listen to rcParams? InlineBackend behavior makes that so complicated :(
        # - Do we ever want to *not* use retina mode at this point?
        dpi = 96
        buffer = io.BytesIO()
        self._figure.savefig(buffer, dpi=dpi * 2, format="png", bbox_inches="tight")
        data = buffer.getvalue()

        scaling = .85
        w, h = self._figure.get_size_inches()
        metadata = {"width": w * dpi * scaling, "height": h * dpi * scaling}
        return data, metadata

    def _setup_data(self, p: Plot) -> None:

        self._data = (
            p._data
            .concat(
                p._facetspec.get("source"),
                p._facetspec.get("variables"),
            )
            .concat(
                p._pairspec.get("source"),
                p._pairspec.get("variables"),
            )
        )

        # TODO concat with mapping spec
        self._layers = []
        for layer in p._layers:
            self._layers.append({
                "data": self._data.concat(layer.get("source"), layer.get("variables")),
                **layer,
            })

    def _setup_figure(self, p: Plot) -> None:

        # --- Parsing the faceting/pairing parameterization to specify figure grid

        # TODO use context manager with theme that has been set
        # TODO (maybe wrap THIS function with context manager; would be cleaner)

        self._subplots = subplots = Subplots(
            p._subplotspec, p._facetspec, p._pairspec, self._data,
        )

        # --- Figure initialization
        figure_kws = {"figsize": getattr(p, "_figsize", None)}  # TODO fix
        self._figure = subplots.init_figure(self.pyplot, figure_kws, p._target)

        # --- Figure annotation
        for sub in subplots:
            ax = sub["ax"]
            for axis in "xy":
                axis_key = sub[axis]
                # TODO Should we make it possible to use only one x/y label for
                # all rows/columns in a faceted plot? Maybe using sub{axis}label,
                # although the alignments of the labels from that method leaves
                # something to be desired (in terms of how it defines 'centered').
                names = [
                    self._data.names.get(axis_key),
                    *[layer["data"].names.get(axis_key) for layer in self._layers],
                ]
                label = next((name for name in names if name is not None), None)
                ax.set(**{f"{axis}label": label})

                axis_obj = getattr(ax, f"{axis}axis")
                visible_side = {"x": "bottom", "y": "left"}.get(axis)
                show_axis_label = (
                    sub[visible_side]
                    or axis in p._pairspec and bool(p._pairspec.get("wrap"))
                    or not p._pairspec.get("cartesian", True)
                )
                axis_obj.get_label().set_visible(show_axis_label)
                show_tick_labels = (
                    show_axis_label
                    or p._subplotspec.get(f"share{axis}") not in (
                        True, "all", {"x": "col", "y": "row"}[axis]
                    )
                )
                plt.setp(axis_obj.get_majorticklabels(), visible=show_tick_labels)
                plt.setp(axis_obj.get_minorticklabels(), visible=show_tick_labels)

            # TODO title template should be configurable
            # TODO Also we want right-side titles for row facets in most cases
            # TODO should configure() accept a title= kwarg (for single subplot plots)?
            # Let's have what we currently call "margin titles" but properly using the
            # ax.set_title interface (see my gist)
            title_parts = []
            for dim in ["row", "col"]:
                if sub[dim] is not None:
                    name = self._data.names.get(dim, f"_{dim}_")
                    title_parts.append(f"{name} = {sub[dim]}")

            has_col = sub["col"] is not None
            has_row = sub["row"] is not None
            show_title = (
                has_col and has_row
                or (has_col or has_row) and p._facetspec.get("wrap")
                or (has_col and sub["top"])
                # TODO or has_row and sub["right"] and <right titles>
                or has_row  # TODO and not <right titles>
            )
            if title_parts:
                title = " | ".join(title_parts)
                title_text = ax.set_title(title)
                title_text.set_visible(show_title)

    def _setup_scales(self, p: Plot) -> None:

        # Identify all of the variables that will be used at some point in the plot
        df = self._data.frame
        variables = list(df)
        for layer in self._layers:
            variables.extend(c for c in layer["data"].frame if c not in variables)

        # Catch cases where a variable is explicitly scaled but has no data,
        # which is *likely* to be a user error (i.e. a typo or mis-specified plot).
        # It's possible we'd want to allow the coordinate axes to be scaled without
        # data, which would let the Plot interface be used to set up an empty figure.
        # So we could revisit this if that seems useful.
        undefined = set(p._scales) - set(variables)
        if undefined:
            err = f"No data found for variable(s) with explicit scale: {undefined}"
            raise RuntimeError(err)  # FIXME:PlotSpecError

        self._scales = {}

        for var in variables:

            # Get the data all the distinct appearances of this variable.
            var_data = pd.concat([
                df.get(var),
                # Only use variables that are *added* at the layer-level
                *(x["data"].frame.get(var)
                  for x in self._layers if var in x["variables"])
            ], axis=1)

            # Determine whether this is an coordinate variable
            # (i.e., x/y, paired x/y, or derivative such as xmax)
            m = re.match(r"^(?P<prefix>(?P<axis>[x|y])\d*).*", var)
            if m is None:
                axis = None
            else:
                var = m.group("prefix")
                axis = m.group("axis")

            # Get the scale object, tracking whether it was explicitly set
            var_values = var_data.stack()
            if var in p._scales:
                scale = p._scales[var]
                scale.type_declared = True
            else:
                scale = get_default_scale(var_values)
                scale.type_declared = False

            # Initialize the data-dependent parameters of the scale
            # Note that this returns a copy and does not mutate the original
            # This dictionary is used by the semantic mappings
            self._scales[var] = scale.setup(var_values)

            # The mappings are always shared across subplots, but the coordinate
            # scaling can be independent (i.e. with share{x/y} = False).
            # So the coordinate scale setup is more complicated, and the rest of the
            # code is only used for coordinate scales.
            if axis is None:
                continue

            share_state = self._subplots.subplot_spec[f"share{axis}"]

            # Shared categorical axes are broken on matplotlib<3.4.0.
            # https://github.com/matplotlib/matplotlib/pull/18308
            # This only affects us when sharing *paired* axes.
            # While it would be possible to hack a workaround together,
            # this is a novel/niche behavior, so we will just raise.
            if LooseVersion(mpl.__version__) < "3.4.0":
                paired_axis = axis in p._pairspec
                cat_scale = self._scales[var].scale_type == "categorical"
                ok_dim = {"x": "col", "y": "row"}[axis]
                shared_axes = share_state not in [False, "none", ok_dim]
                if paired_axis and cat_scale and shared_axes:
                    err = "Sharing paired categorical axes requires matplotlib>=3.4.0"
                    raise RuntimeError(err)

            # Loop over every subplot and assign its scale if it's not in the axis cache
            for subplot in self._subplots:

                # This happens when Plot.pair was used
                if subplot[axis] != var:
                    continue

                axis_obj = getattr(subplot["ax"], f"{axis}axis")
                set_scale_obj(subplot["ax"], axis, scale)

                # Now we need to identify the right data rows to setup the scale with

                # The all-shared case is easiest, every subplot sees all the data
                if share_state in [True, "all"]:
                    axis_scale = scale.setup(var_values, axis_obj)
                    subplot[f"{axis}scale"] = axis_scale

                # Otherwise, we need to setup separate scales for different subplots
                else:
                    # Fully independent axes are easy, we use each subplot's data
                    if share_state in [False, "none"]:
                        subplot_data = self._filter_subplot_data(df, subplot)
                    # Sharing within row/col is more complicated
                    elif share_state in df:
                        subplot_data = df[df[share_state] == subplot[share_state]]
                    else:
                        subplot_data = df

                    # Same operation as above, but using the reduced dataset
                    subplot_values = var_data.loc[subplot_data.index].stack()
                    axis_scale = scale.setup(subplot_values, axis_obj)
                    subplot[f"{axis}scale"] = axis_scale

        # Set default axis scales for when they're not defined at this point
        for subplot in self._subplots:
            ax = subplot["ax"]
            for axis in "xy":
                key = f"{axis}scale"
                if key not in subplot:
                    default_scale = scale_factory(getattr(ax, f"get_{key}")(), axis)
                    # TODO should we also infer categories / datetime units?
                    subplot[key] = NumericScale(default_scale, None)

    def _setup_mappings(self, p: Plot) -> None:

        semantic_vars: list[str]
        mapping: SemanticMapping

        variables = list(self._data.frame)  # TODO abstract this?
        for layer in self._layers:
            variables.extend(c for c in layer["data"].frame if c not in variables)
        semantic_vars = [v for v in variables if v in SEMANTICS]

        self._mappings = {}
        for var in semantic_vars:
            semantic = p._semantics.get(var) or SEMANTICS[var]

            all_values = pd.concat([
                self._data.frame.get(var),
                # Only use variables that are *added* at the layer-level
                *(x["data"].frame.get(var)
                  for x in self._layers if var in x["variables"])
            ], axis=1).stack()

            if var in self._scales:
                scale = self._scales[var]
                scale.type_declared = True
            else:
                scale = get_default_scale(all_values)
                scale.type_declared = False

            if isinstance(scale, IdentityScale):
                # We may not need this dummy mapping, if we can consistently
                # use Mark.resolve to pull values out of data if not defined in mappings
                # Not doing that now because it breaks some tests, but seems to work.
                mapping = IdentityMapping(semantic._standardize_values)
            else:
                mapping = semantic.setup(all_values, scale)
            self._mappings[var] = mapping

    def _plot_layer(
        self,
        p: Plot,
        layer: dict[str, Any],  # TODO layer should be a TypedDict
    ) -> None:

        default_grouping_vars = ["col", "row", "group"]  # TODO where best to define?

        data = layer["data"]
        mark = layer["mark"]
        stat = layer["stat"]

        pair_variables = p._pairspec.get("structure", {})

        full_df = data.frame
        for subplots, df, scales in self._generate_pairings(full_df, pair_variables):

            orient = layer["orient"] or mark._infer_orient(scales)

            with (
                mark.use(self._mappings, orient)
                # TODO this doesn't work if stat is None
                # stat.use(mappings=self._mappings, orient=orient),
            ):

                df = self._scale_coords(subplots, df)

                if stat is not None:
                    grouping_vars = stat.grouping_vars + default_grouping_vars
                    df = self._apply_stat(df, grouping_vars, stat, orient)

                df = mark._adjust(df)

                df = self._unscale_coords(subplots, df)

                grouping_vars = mark.grouping_vars + default_grouping_vars
                split_generator = self._setup_split_generator(
                    grouping_vars, df, subplots
                )

                mark._plot(split_generator)

        with mark.use(self._mappings, None):  # TODO will we ever need orient?
            self._update_legend_contents(mark, data)

    def _apply_stat(
        self,
        df: DataFrame,
        grouping_vars: list[str],
        stat: Stat,
        orient: Literal["x", "y"],
    ) -> DataFrame:

        stat.setup(df, orient)  # TODO pass scales here?

        # TODO how can we special-case fast aggregations? (i.e. mean, std, etc.)
        # IDEA: have Stat identify as an aggregator? (Through Mixin or attribute)
        # e.g. if stat.aggregates ...
        stat_grouping_vars = [var for var in grouping_vars if var in df]
        # TODO I don't think we always want to group by the default orient axis?
        # Better to have the Stat declare when it wants that to happen
        if orient not in stat_grouping_vars:
            stat_grouping_vars.append(orient)

        # TODO rewrite this whole thing, I think we just need to avoid groupby/apply
        df = (
            df
            .groupby(stat_grouping_vars)
            .apply(stat)
        )
        # TODO next because of https://github.com/pandas-dev/pandas/issues/34809
        for var in stat_grouping_vars:
            if var in df.index.names:
                df = (
                    df
                    .drop(var, axis=1, errors="ignore")
                    .reset_index(var)
                )
        df = df.reset_index(drop=True)  # TODO not always needed, can we limit?
        return df

    def _scale_coords(
        self,
        subplots: list[dict],  # TODO retype with a SubplotSpec or similar
        df: DataFrame,
    ) -> DataFrame:

        coord_cols = [c for c in df if re.match(r"^[xy]\D*$", c)]
        out_df = (
            df
            .copy(deep=False)
            .drop(coord_cols, axis=1)
            .reindex(df.columns, axis=1)  # So unscaled columns retain their place
        )

        for subplot in subplots:
            axes_df = self._filter_subplot_data(df, subplot)[coord_cols]
            with pd.option_context("mode.use_inf_as_null", True):
                axes_df = axes_df.dropna()  # TODO always wanted?
            for var, values in axes_df.items():
                axis = var[0]
                scale = subplot[f"{axis}scale"]
                axis_obj = getattr(subplot["ax"], f"{axis}axis")
                out_df.loc[values.index, var] = scale.forward(values, axis_obj)

        return out_df

    def _unscale_coords(
        self,
        subplots: list[dict],  # TODO retype with a SubplotSpec or similar
        df: DataFrame
    ) -> DataFrame:

        coord_cols = [c for c in df if re.match(r"^[xy]\D*$", c)]
        out_df = (
            df
            .drop(coord_cols, axis=1)
            .copy(deep=False)
            .reindex(df.columns, axis=1)  # So unscaled columns retain their place
        )

        for subplot in subplots:
            axes_df = self._filter_subplot_data(df, subplot)[coord_cols]
            for var, values in axes_df.items():
                scale = subplot[f"{var[0]}scale"]
                out_df.loc[values.index, var] = scale.reverse(axes_df[var])

        return out_df

    def _generate_pairings(
        self,
        df: DataFrame,
        pair_variables: dict,
    ) -> Generator[
        tuple[list[dict], DataFrame, dict[str, Scale]], None, None
    ]:
        # TODO retype return with SubplotSpec or similar

        if not pair_variables:
            # TODO casting to list because subplots below is a list
            # Maybe a cleaner way to do this?
            yield list(self._subplots), df, self._scales
            return

        iter_axes = itertools.product(*[
            pair_variables.get(axis, [None]) for axis in "xy"
        ])

        for x, y in iter_axes:

            subplots = []
            for sub in self._subplots:
                if (x is None or sub["x"] == x) and (y is None or sub["y"] == y):
                    subplots.append(sub)

            reassignments = {}
            for axis, prefix in zip("xy", [x, y]):
                if prefix is not None:
                    reassignments.update({
                        # Complex regex business to support e.g. x0max
                        re.sub(rf"^{prefix}(.*)$", rf"{axis}\1", col): df[col]
                        for col in df if col.startswith(prefix)
                    })

            scales = {new: self._scales[old.name] for new, old in reassignments.items()}

            yield subplots, df.assign(**reassignments), scales

    def _filter_subplot_data(
        self,
        df: DataFrame,
        subplot: dict,
    ) -> DataFrame:

        keep_rows = pd.Series(True, df.index, dtype=bool)
        for dim in ["col", "row"]:
            if dim in df:
                keep_rows &= df[dim] == subplot[dim]
        return df[keep_rows]

    def _setup_split_generator(
        self,
        grouping_vars: list[str],
        df: DataFrame,
        subplots: list[dict[str, Any]],
    ) -> Callable[[], Generator]:

        allow_empty = False  # TODO will need to recreate previous categorical plots

        grouping_keys = []
        grouping_vars = [
            v for v in grouping_vars if v in df and v not in ["col", "row"]
        ]
        for var in grouping_vars:
            order = self._scales[var].order
            if order is None:
                order = categorical_order(df[var])
            grouping_keys.append(order)

        def split_generator() -> Generator:

            for subplot in subplots:

                axes_df = self._filter_subplot_data(df, subplot)

                subplot_keys = {}
                for dim in ["col", "row"]:
                    if subplot[dim] is not None:
                        subplot_keys[dim] = subplot[dim]

                if not grouping_vars or not any(grouping_keys):
                    yield subplot_keys, axes_df.copy(), subplot["ax"]
                    continue

                grouped_df = axes_df.groupby(grouping_vars, sort=False, as_index=False)

                for key in itertools.product(*grouping_keys):

                    # Pandas fails with singleton tuple inputs
                    pd_key = key[0] if len(key) == 1 else key

                    try:
                        df_subset = grouped_df.get_group(pd_key)
                    except KeyError:
                        # TODO (from initial work on categorical plots refactor)
                        # We are adding this to allow backwards compatability
                        # with the empty artists that old categorical plots would
                        # add (before 0.12), which we may decide to break, in which
                        # case this option could be removed
                        df_subset = axes_df.loc[[]]

                    if df_subset.empty and not allow_empty:
                        continue

                    sub_vars = dict(zip(grouping_vars, key))
                    sub_vars.update(subplot_keys)

                    yield sub_vars, df_subset.copy(), subplot["ax"]

        return split_generator

    def _update_legend_contents(self, mark: Mark, data: PlotData) -> None:
        """Add legend artists / labels for one layer in the plot."""
        legend_vars = data.frame.columns.intersection(self._mappings)

        # First pass: Identify the values that will be shown for each variable
        schema: list[tuple[
            tuple[str | None, str | int], list[str], tuple[list, list[str]]
        ]] = []
        schema = []
        for var in legend_vars:
            var_legend = self._mappings[var].legend
            if var_legend is not None:
                values, labels = var_legend
                for (_, part_id), part_vars, _ in schema:
                    if data.ids[var] == part_id:
                        # Allow multiple plot semantics to represent same data variable
                        part_vars.append(var)
                        break
                else:
                    entry = (data.names[var], data.ids[var]), [var], (values, labels)
                    schema.append(entry)

        # Second pass, generate an artist corresponding to each value
        contents = []
        for key, variables, (values, labels) in schema:
            artists = []
            for val in values:
                artists.append(mark._legend_artist(variables, val))
            contents.append((key, artists, labels))

        self._legend_contents.extend(contents)

    def _make_legend(self) -> None:
        """Create the legend artist(s) and add onto the figure."""
        # Combine artists representing same information across layers
        # Input list has an entry for each distinct variable in each layer
        # Output dict has an entry for each distinct variable
        merged_contents: dict[
            tuple[str | None, str | int], tuple[list[Artist], list[str]],
        ] = {}
        for key, artists, labels in self._legend_contents:
            # Key is (name, id); we need the id to resolve variable uniqueness,
            # but will need the name in the next step to title the legend
            if key in merged_contents:
                # Copy so inplace updates don't propagate back to legend_contents
                existing_artists = merged_contents[key][0].copy()
                for i, artist in enumerate(existing_artists):
                    # Matplotlib accepts a tuple of artists and will overlay them
                    if isinstance(artist, tuple):
                        artist += artist[i],
                    else:
                        artist = artist, artists[i]
                    # Update list that is a value in the merged_contents dict in place
                    existing_artists[i] = artist
            else:
                merged_contents[key] = artists, labels

        base_legend = None
        for (name, _), (handles, labels) in merged_contents.items():

            legend = mpl.legend.Legend(
                self._figure,
                handles,
                labels,
                title=name,  # TODO don't show "None" as title
                loc="upper right",
                # bbox_to_anchor=(.98, .98),
            )

            # TODO: This is an illegal hack accessing private attributes on the legend
            # We need to sort out how we are going to handle this given that lack of a
            # proper API to do things like position legends relative to each other
            if base_legend:
                base_legend._legend_box._children.extend(legend._legend_box._children)
            else:
                base_legend = legend
                self._figure.legends.append(legend)
