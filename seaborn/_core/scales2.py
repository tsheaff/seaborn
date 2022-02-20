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

    scale_type: ClassVar[str] = "categorical"  # TODO

    def setup(
        self,
        data: Series,
        prop: Property | None = None,
        axis: Axis | None = None,
    ) -> Scale:

        transform, inverse = _make_identity_transforms()

        # TODO plug into axis

        forward_pipe = [
            # np.vectorize(format),  # TODO ensure stringification
            prop.get_mapping(self, data),
            # TODO how to handle color representation consistency?
        ]

        inverse_pipe = []

        matplotlib_scale = mpl.scale.LinearScale(data.name)

        return Scale(forward_pipe, inverse_pipe, "categorical", matplotlib_scale)


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
        transform, inverse = self.get_transform()

        new._transform = transform

        forward_pipe = [
            pd.to_numeric,
            transform,
            prop.get_norm(new, data),
            prop.get_mapping(new, data)
        ]

        inverse_pipe = [inverse]

        # matplotlib_scale = mpl.scale.LinearScale(data.name)
        matplotlib_scale = mpl.scale.FuncScale(data.name, (transform, inverse))

        return Scale(forward_pipe, inverse_pipe, "numeric", matplotlib_scale)

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


class Sequential(Continuous):

    ...


class Diverging(Continuous):

    ...


class Qualitative(Nominal):

    ...


class Binned(ScaleSpec):
    # Needed? Or handle this at layer (in stat or as param, eg binning=)
    ...


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
        return log(x) - log(1 - x)

    def expit(x):
        return exp(x) / (1 + exp(x))

    return logit, expit


def _make_log_transforms(base=None):

    if base is None:
        return np.log, np.exp
    elif base == 2:
        return np.log2, partial(np.power, 2)
    elif base == 10:
        return np.log10, partial(np.power, 10)
    else:
        def forward(x):
            return np.log(x) / np.log(base)
        return forward, partial(np.power, base)


def _make_symlog_transforms(c=1, base=10):

    # From https://iopscience.iop.org/article/10.1088/0957-0233/24/2/027001

    # Note: currently not using base because we only get
    # one parameter from the string, and are using c (this is consistent with d3)

    log, exp = _make_log_transforms(base)

    def symlog(x):
        return np.sign(x) * log(1 + np.abs(np.divide(x, c)))

    def symexp(x):
        return np.sign(x) * c * (exp(np.abs(x)) - 1)

    return symlog, symexp


def _make_sqrt_transforms():

    return np.sqrt, np.square


def _make_power_transforms(exp):

    def forward(x):
        return np.power(x, exp)

    def inverse(x):
        return np.power(x, 1 / exp)

    return forward, inverse
