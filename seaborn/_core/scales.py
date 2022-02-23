from __future__ import annotations
from copy import copy
from dataclasses import dataclass
from functools import partial
from typing import ClassVar

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.axis import Axis

from seaborn._core.rules import categorical_order
from seaborn._compat import set_scale_obj

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any, Callable
    from matplotlib.scale import ScaleBase
    from pandas import Series
    from seaborn._core.properties import Property


class Scale:

    def __init__(
        self,
        forward_pipe,
        inverse_pipe=None,
        scale_type=None,
        matplotlib_scale=None,
    ):

        self.forward_pipe = forward_pipe
        self.inverse_pipe = inverse_pipe
        self.scale_type = scale_type
        self.matplotlib_scale = matplotlib_scale

        # TODO need to make this work
        self.order = None

    def __call__(self, data):

        return self._apply_pipeline(data, self.forward_pipe)

    def _apply_pipeline(self, data, pipeline):

        for func in pipeline:
            if func is not None:
                data = func(data)
        return data

    def invert_transform(self, data):
        assert self.inverse_pipe is not None  # TODO raise or no-op?
        return self._apply_pipeline(data, self.inverse_pipe)

    def get_matplotlib_scale(self):
        # TODO is this the best approach? Should this be "lazy"?
        return self.matplotlib_scale


@dataclass
class ScaleSpec:

    ...

    # TODO have Scale define width (/height?) (using data?), so e.g. nominal scale sets
    # width=1, continuous scale sets width min(diff(unique(data))), etc.


@dataclass
class Nominal(ScaleSpec):

    # Categorical (convert to strings), un-sortable
    values: str | list | dict | None = None
    order: list | None = None

    scale_type: ClassVar[str] = "categorical"  # TODO

    def setup(
        self,
        data: Series,
        prop: Property | None = None,
        axis: Axis | None = None,
    ) -> Scale:

        class CatScale(mpl.scale.LinearScale):
            # TODO turn this into a real thing I guess
            def set_default_locators_and_formatters(self, axis):
                pass

        mpl_scale = CatScale(data.name)
        if axis is None:
            axis = PseudoAxis(mpl_scale)

        # TODO flexibility over format() which isn't great for numbers / dates
        stringify = np.vectorize(format)

        units_seed = stringify(categorical_order(data, self.order))
        axis.update_units(units_seed)

        # TODO define this more centrally
        def convert_units(x):
            # TODO only do this with explicit order?
            # (But also category dtype?)
            keep = np.in1d(x, units_seed)
            out = np.full(x.shape, np.nan)
            out[keep] = axis.convert_units(x[keep])
            return out

        forward_pipe = [
            stringify,
            convert_units,
            prop.get_mapping(self, data),
            # TODO how to handle color representation consistency?
        ]

        inverse_pipe = []

        return Scale(forward_pipe, inverse_pipe, "categorical", mpl_scale)


@dataclass
class Ordinal(ScaleSpec):
    # Categorical (convert to strings), sortable
    ...


@dataclass
class Discrete(ScaleSpec):
    # Numeric, integral, can skip ticks/ticklabels
    palette: str | list | dict | None = None
    order: list | None = None
    # TODO other params


@dataclass
class Continuous(ScaleSpec):

    values: tuple[float, float] | None = None
    norm: tuple[float | None, float | None] | None = None
    transform: str | tuple[Callable, Callable] | None = None
    outside: Literal["keep", "drop", "clip"] = "keep"
    # TODO other params

    # TODO needed for Mark._infer_orient
    # But maybe that should do an isinstance check?
    scale_type: ClassVar[str] = "numeric"

    def tick(self, count=None, *, every=None, at=None, format=None):
        # Other ideas ... between?
        # How to minor ticks? I am fine with minor ticks never getting labels
        # so it is just a matter or specifing a) you want them and b) how many?
        # Unlike with ticks, knowing how many minor ticks in each interval suffices.
        # So I guess we just need a good parameter name?
        # Do we want to allow tick appearance parameters here?
        # What about direction? Tick on alternate axis?
        # And specific tick label values? Only allow for categorical scales?
        ...

    # How to *allow* use of more complex third party objects? It seems shortsighted
    # not to maintain capabilities afforded by Scale / Ticker / Locator / UnitData,
    # despite the complexities of that API.
    # def using(self, scale: mpl.scale.ScaleBase) ?

    def setup(
        self,
        data: Series,
        prop: Property | None = None,
        axis: Axis | None = None,
    ) -> Scale:

        new = copy(self)
        forward, inverse = self.get_transform()

        new._transform = forward

        # matplotlib_scale = mpl.scale.LinearScale(data.name)
        mpl_scale = mpl.scale.FuncScale(data.name, (forward, inverse))

        if axis is None:
            axis = PseudoAxis(mpl_scale)
            axis.update_units(data)

        forward_pipe = [
            axis.convert_units,
            forward,
            prop.get_norm(new, data),
            prop.get_mapping(new, data)
        ]

        inverse_pipe = [inverse]

        return Scale(forward_pipe, inverse_pipe, "numeric", mpl_scale)

    def get_transform(self):

        arg = self.transform

        def get_param(method, default):
            if arg == method:
                return default
            return float(arg[len(method):])

        if arg is None:
            return _make_identity_transforms()
        elif isinstance(arg, tuple):
            return arg
        elif isinstance(arg, str):
            if arg == "ln":
                return _make_log_transforms()
            elif arg == "logit":
                base = get_param("logit", 10)
                return _make_logit_transforms(base)
            elif arg.startswith("log"):
                base = get_param("log", 10)
                return _make_log_transforms(base)
            elif arg.startswith("symlog"):
                c = get_param("symlog", 1)
                return _make_symlog_transforms(c)
            elif arg.startswith("pow"):
                exp = get_param("pow", 2)
                return _make_power_transforms(exp)
            elif arg == "sqrt":
                return _make_sqrt_transforms()
            else:
                # TODO useful error message
                raise ValueError()


# ----------------------------------------------------------------------------------- #

# TODO best way to handle datetime(s)?

class Temporal(ScaleSpec):
    ...


class Calendric(ScaleSpec):
    ...


class Binned(ScaleSpec):
    # Needed? Or handle this at layer (in stat or as param, eg binning=)
    ...


# TODO any need for color-specific scales?


class Sequential(Continuous):
    ...


class Diverging(Continuous):
    ...
    # TODO alt approach is to have Continuous.center()


class Qualitative(Nominal):
    ...


# ----------------------------------------------------------------------------------- #


class PseudoAxis:
    """
    Internal class implementing minimal interface equivalent to matplotlib Axis.

    Coordinate variables are typically scaled by attaching the Axis object from
    the figure where the plot will end up. Matplotlib has no similar concept of
    and axis for the other mappable variables (color, etc.), but to simplify the
    code, this object acts like an Axis and can be used to scale other variables.

    """
    axis_name = ""  # TODO Needs real value? Just used for x/y logic in matplotlib

    def __init__(self, scale):

        self.converter = None
        self.units = None
        self.scale = scale
        self.major = mpl.axis.Ticker()

        scale.set_default_locators_and_formatters(self)
        # self.set_default_intervals()  TODO mock?

    def set_view_interval(self, vmin, vmax):
        # TODO this gets called when setting DateTime units,
        # but we may not need it to do anything
        self._view_interval = vmin, vmax

    def get_view_interval(self):
        return self._view_interval

    # TODO do we want to distinguish view/data intervals? e.g. for a legend
    # we probably want to represent the full range of the data values, but
    # still norm the colormap. If so, we'll need to track data range separately
    # from the norm, which we currently don't do.

    def set_data_interval(self, vmin, vmax):
        self._data_interval = vmin, vmax

    def get_data_interval(self):
        return self._data_interval

    def get_tick_space(self):
        # TODO how to do this in a configurable / auto way?
        # Would be cool to have legend density adapt to figure size, etc.
        return 5

    def set_major_locator(self, locator):
        self.major.locator = locator
        locator.set_axis(self)

    def set_major_formatter(self, formatter):
        # TODO matplotlib method does more handling (e.g. to set w/format str)
        self.major.formatter = formatter
        formatter.set_axis(self)

    def set_minor_locator(self, locator):
        pass

    def set_minor_formatter(self, formatter):
        pass

    def set_units(self, units):
        self.units = units

    def update_units(self, x):
        """Pass units to the internal converter, potentially updating its mapping."""
        self.converter = mpl.units.registry.get_converter(x)
        if self.converter is not None:
            self.converter.default_units(x, self)

            info = self.converter.axisinfo(self.units, self)

            if info is None:
                return
            if info.majloc is not None:
                # TODO matplotlib method has more conditions here; are they needed?
                self.set_major_locator(info.majloc)
            if info.majfmt is not None:
                self.set_major_formatter(info.majfmt)

            # TODO this is in matplotlib method; do we need this?
            # self.set_default_intervals()

    def convert_units(self, x):
        """Return a numeric representation of the input data."""
        if self.converter is None:
            return x
        return self.converter.convert(x, self.units, self)


# -


def infer_scale_type(var: str, data: Series, arg: Any) -> Scale:

    if arg is None:
        # Note that we also want None in Plot.scale to mean identity
        # Perhaps have a separate function for inferring just from data?
        ...


def _make_identity_transforms():

    def identity(x):
        return x

    return identity, identity


def _make_logit_transforms(base=None):

    log, exp = _make_log_transforms(base)

    def logit(x):
        with np.errstate(invalid="ignore", divide="ignore"):
            return log(x) - log(1 - x)

    def expit(x):
        with np.errstate(invalid="ignore", divide="ignore"):
            return exp(x) / (1 + exp(x))

    return logit, expit


def _make_log_transforms(base=None):

    if base is None:
        fs = np.log, np.exp
    elif base == 2:
        fs = np.log2, partial(np.power, 2)
    elif base == 10:
        fs = np.log10, partial(np.power, 10)
    else:
        def forward(x):
            return np.log(x) / np.log(base)
        fs = forward, partial(np.power, base)

    def log(x):
        with np.errstate(invalid="ignore", divide="ignore"):
            return fs[0](x)

    def exp(x):
        with np.errstate(invalid="ignore", divide="ignore"):
            return fs[1](x)

    return log, exp


def _make_symlog_transforms(c=1, base=10):

    # From https://iopscience.iop.org/article/10.1088/0957-0233/24/2/027001

    # Note: currently not using base because we only get
    # one parameter from the string, and are using c (this is consistent with d3)

    log, exp = _make_log_transforms(base)

    def symlog(x):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.sign(x) * log(1 + np.abs(np.divide(x, c)))

    def symexp(x):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.sign(x) * c * (exp(np.abs(x)) - 1)

    return symlog, symexp


def _make_sqrt_transforms():

    def sqrt(x):
        return np.sign(x) * np.sqrt(np.abs(x))

    def square(x):
        return np.sign(x) * np.square(x)

    return sqrt, square


def _make_power_transforms(exp):

    def forward(x):
        return np.sign(x) * np.power(np.abs(x), exp)

    def inverse(x):
        return np.sign(x) * np.power(np.abs(x), 1 / exp)

    return forward, inverse
