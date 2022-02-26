from __future__ import annotations
import itertools

import numpy as np
import matplotlib as mpl

from seaborn._core.scales import ScaleSpec, Nominal, Continuous
from seaborn._core.rules import categorical_order, variable_type
from seaborn.palettes import QUAL_PALETTES, color_palette, blend_palette
from seaborn.utils import get_color_cycle

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pandas import Series
    from matplotlib.markers import MarkerStyle


class Property:

    @property
    def default_range(self) -> tuple[float, float]:
        return self._default_range

    def default_scale(self, data: Series) -> ScaleSpec:
        # TODO use Boolean if we add that as a scale
        # TODO how will this handle data with units that can be treated as numeric?
        var_type = variable_type(data, boolean_type="categorical")
        if var_type == "numeric":
            return Continuous()
        # TODO Temporal?
        else:
            return Nominal()

    def infer_scale(self, arg, data):
        # TODO what is best base-level default?
        var_type = variable_type(data)

        # TODO put these somewhere external for validation
        # TODO putting this here won't pick it up if subclasses define infer_scale
        # (e.g. color). How best to handle that? One option is to call super after
        # handling property-specific possibilities (e.g. for color check that the
        # arg is not a valid palette name) but that could get tricky.
        trans_args = ["log", "symlog", "logit", "pow", "sqrt"]
        if isinstance(arg, str) and any(arg.startswith(k) for k in trans_args):
            return Continuous(transform=arg)

        if var_type == "categorical":
            return Nominal(arg)
        else:
            return Continuous(arg)

    def get_norm(self, scale, data):
        return None

    def get_mapping(self, scale, data):
        return None


class NormableProperty(Property):

    def get_norm(self, scale, data):
        # TODO this should be at a base class level? But maybe not the lowest?

        if isinstance(scale.norm, tuple):
            vmin, vmax = scale._transform(scale.norm)

            # TODO norm as matplotlib object?

        else:
            vmin, vmax = scale._transform((data.min(), data.max()))

        # TODO use this object or return a closure over vmin/vmax?
        norm = mpl.colors.Normalize(vmin, vmax)

        return norm


class SizedProperty(NormableProperty):

    # TODO pass default range to constructor and avoid defining a bunch of subclasses?
    _default_range: tuple[float, float] = (0, 1)

    def get_mapping(self, scale, data):

        if scale.values is None:
            vmin, vmax = self.default_range
        else:
            vmin, vmax = scale.values

        def f(x):
            return x * (vmax - vmin) + vmin

        return f


class Coordinate(Property):

    _default_range = None


class ObjectProperty(Property):
    # TODO better name this is unclear!

    def infer_scale(self, arg, data):

        return Nominal(arg)

    def get_mapping(self, scale, data):

        levels = categorical_order(data)
        n = len(levels)

        if isinstance(scale.values, dict):
            # self._check_dict_not_missing_levels(levels, values)
            # TODO where to ensure that dict values have consistent representation?
            values = [scale.values[x] for x in levels]
        elif isinstance(scale.values, list):
            # colors = self._ensure_list_not_too_short(levels, values)
            # TODO check not too long also?
            values = scale.values
        elif scale.values is None:
            values = self._default_values(n)
        else:
            # TODO nice error message
            assert False, values

        values = self._standardize_values(values)

        def mapping(x):
            return [values[ix] for ix in x.astype(np.intp)]

        return mapping

    def _default_values(self, n):
        raise NotImplementedError()

    def _standardize_values(self, values):

        return values


class Marker(ObjectProperty):

    # TODO should we have named marker "palettes"? (e.g. see d3 options)

    # TODO will need abstraction to share with LineStyle, etc.

    # TODO need some sort of "require_scale" functionality
    # to raise when we get the wrong kind explicitly specified

    def _standardize_values(self, values):

        return [mpl.markers.MarkerStyle(x) for x in values]

    def _default_values(self, n: int) -> list[MarkerStyle]:
        """Build an arbitrarily long list of unique marker styles for points.

        Parameters
        ----------
        n : int
            Number of unique marker specs to generate.

        Returns
        -------
        markers : list of string or tuples
            Values for defining :class:`matplotlib.markers.MarkerStyle` objects.
            All markers will be filled.

        """
        # Start with marker specs that are well distinguishable
        markers = [
            "o",
            "X",
            (4, 0, 45),
            "P",
            (4, 0, 0),
            (4, 1, 0),
            "^",
            (4, 1, 45),
            "v",
        ]

        # Now generate more from regular polygons of increasing order
        s = 5
        while len(markers) < n:
            a = 360 / (s + 1) / 2
            markers.extend([
                (s + 1, 1, a),
                (s + 1, 0, a),
                (s, 1, 0),
                (s, 0, 0),
            ])
            s += 1

        markers = [mpl.markers.MarkerStyle(m) for m in markers[:n]]

        return markers


class LineStyle(ObjectProperty):

    def _default_values(self, n: int):  # -> list[DashPatternWithOffset]:
        """Build an arbitrarily long list of unique dash styles for lines.

        Parameters
        ----------
        n : int
            Number of unique dash specs to generate.

        Returns
        -------
        dashes : list of strings or tuples
            Valid arguments for the ``dashes`` parameter on
            :class:`matplotlib.lines.Line2D`. The first spec is a solid
            line (``""``), the remainder are sequences of long and short
            dashes.

        """
        # Start with dash specs that are well distinguishable
        dashes = [  # TODO : list[str | DashPattern] = [
            "-",  # TODO do we need to handle this elsewhere for backcompat?
            (4, 1.5),
            (1, 1),
            (3, 1.25, 1.5, 1.25),
            (5, 1, 1, 1),
        ]

        # Now programmatically build as many as we need
        p = 3
        while len(dashes) < n:

            # Take combinations of long and short dashes
            a = itertools.combinations_with_replacement([3, 1.25], p)
            b = itertools.combinations_with_replacement([4, 1], p)

            # Interleave the combinations, reversing one of the streams
            segment_list = itertools.chain(*zip(
                list(a)[1:-1][::-1],
                list(b)[1:-1]
            ))

            # Now insert the gaps
            for segments in segment_list:
                gap = min(segments)
                spec = tuple(itertools.chain(*((seg, gap) for seg in segments)))
                dashes.append(spec)

            p += 1

        return self._standardize_values(dashes)

    def _standardize_values(self, values):
        """Standardize values as dash pattern (with offset)."""
        return [self._get_dash_pattern(x) for x in values]

    @staticmethod
    def _get_dash_pattern(style: str | DashPattern) -> DashPatternWithOffset:
        """Convert linestyle arguments to dash pattern with offset."""
        # Copied and modified from Matplotlib 3.4
        # go from short hand -> full strings
        ls_mapper = {'-': 'solid', '--': 'dashed', '-.': 'dashdot', ':': 'dotted'}
        if isinstance(style, str):
            style = ls_mapper.get(style, style)
            # un-dashed styles
            if style in ['solid', 'none', 'None']:
                offset = 0
                dashes = None
            # dashed styles
            elif style in ['dashed', 'dashdot', 'dotted']:
                offset = 0
                dashes = tuple(mpl.rcParams[f'lines.{style}_pattern'])

        elif isinstance(style, tuple):
            if len(style) > 1 and isinstance(style[1], tuple):
                offset, dashes = style
            elif len(style) > 1 and style[1] is None:
                offset, dashes = style
            else:
                offset = 0
                dashes = style
        else:
            raise ValueError(f'Unrecognized linestyle: {style}')

        # normalize offset to be positive and shorter than the dash cycle
        if dashes is not None:
            dsum = sum(dashes)
            if dsum:
                offset %= dsum

        return offset, dashes


class Color(NormableProperty):

    def infer_scale(self, arg, data) -> ScaleSpec:

        # TODO do color standardization on dict / list values?
        if isinstance(arg, (dict, list)):
            return Nominal(arg)

        if isinstance(arg, tuple):
            return Continuous(arg)

        if callable(arg):
            return Continuous(arg)

        # TODO Do we accept str like "log", "pow", etc. for semantics?

        # TODO what about
        # - Temporal? (i.e. datetime)
        # - Boolean?

        assert isinstance(arg, str)  # TODO sanity check

        var_type = (
            "categorical" if arg in QUAL_PALETTES
            else variable_type(data, boolean_type="categorical")
        )

        if var_type == "categorical":
            return Nominal(arg)

        if var_type == "numeric":
            return Continuous(arg)

        # TODO just to see when we get here
        assert False

    def _get_categorical_mapping(self, scale, data):

        levels = categorical_order(data)
        n = len(levels)
        values = scale.values

        if isinstance(values, dict):
            # self._check_dict_not_missing_levels(levels, values)
            # TODO where to ensure that dict values have consistent representation?
            colors = [values[x] for x in levels]
        else:
            if values is None:
                if n <= len(get_color_cycle()):
                    # Use current (global) default palette
                    colors = color_palette(n_colors=n)
                else:
                    colors = color_palette("husl", n)
            elif isinstance(values, list):
                # colors = self._ensure_list_not_too_short(levels, values)
                # TODO check not too long also?
                colors = color_palette(values)
            else:
                colors = color_palette(values, n)

        return colors

    def get_mapping(self, scale, data):

        # TODO what is best way to do this conditional?
        if isinstance(scale, Nominal):
            colors = self._get_categorical_mapping(scale, data)

            def mapping(x):
                return np.take(colors, x.astype(np.intp), axis=0)

        elif scale.values is None:
            # TODO data-dependent default type
            # (Or should caller dispatch to function / dictionary mapping?)
            mapping = color_palette("ch:", as_cmap=True)
        elif isinstance(scale.values, tuple):
            mapping = blend_palette(scale.values, as_cmap=True)
        elif isinstance(scale.values, str):
            # TODO data-dependent return type?
            # TODO for matplotlib colormaps this will clip, which is different behavior
            mapping = color_palette(scale.values, as_cmap=True)

        # TODO just during dev
        else:
            assert False

        # TODO figure out better way to do this, maybe in color_palette?
        # Also note that this does not preserve alpha channels when given
        # as part of the range values, which we want.
        def _mapping(x):
            return mapping(x)[:, :3]

        return _mapping


class PointSize(SizedProperty):
    _default_range = 2, 8


class LineWidth(SizedProperty):
    @property
    def default_range(self) -> tuple[float, float]:
        base = mpl.rcParams["lines.linewidth"]
        return base * .5, base * 2


# TODO should these be instances or classes?
PROPERTIES = {
    "x": Coordinate(),
    "y": Coordinate(),
    "color": Color(),
    "fillcolor": Color(),
    "edgecolor": Color(),
    "alpha": ...,
    "fillalpha": ...,
    "edgealpha": ...,
    "fill": ...,
    "marker": Marker(),
    "linestyle": LineStyle(),
    "pointsize": PointSize(),
    "linewidth": LineWidth(),
    "edgewidth": ...
}
