from __future__ import annotations

import matplotlib as mpl

from seaborn._core.scales2 import ScaleSpec, Nominal, Continuous
from seaborn._core.rules import categorical_order, variable_type
from seaborn.palettes import QUAL_PALETTES, color_palette, blend_palette


class Property:

    def default_scale(self, data):
        # TODO use Boolean if we add that as a scale
        # TODO how will this handle data with units that can be treated as numeric?
        var_type = variable_type(data, boolean_type="categorical")
        if var_type == "numeric":
            return Continuous()
        # TODO Temporal?
        else:
            return Nominal()

    def get_norm(self, arg, data, transform):
        return None

    def get_mapping(self, arg, data):
        return None


class Coordinate(Property):

    default_range = None

    def infer_scale(self, arg, data):

        var_type = variable_type(data)

        if var_type == "categorical":
            return Nominal()
        else:
            return Continuous()


class Color(Property):

    def infer_scale(self, arg, data) -> ScaleSpec:

        # TODO do color standardization on dict / list values?
        if isinstance(arg, dict):
            # TODO check_dict_not_missing_levels
            return Nominal(arg)

        if isinstance(arg, list):
            levels = categorical_order(data)
            # TODO ensure_list_not_too_short
            return Nominal(dict(zip(levels, arg)))

        # TODO Do we accept str like "log", "pow", etc. for semantics?

        # TODO tuple of colors -> blend colormap

        # TODO what about
        # - Temporal? (i.e. datetime)
        # - Boolean?

        # TODO otherwise assume we have the name of a palette?
        # what about matplotib colormap? should Continuous accept a generic
        # func that goes from [0, 1] -> range value?

        assert isinstance(arg, str)  # TODO sanity check

        var_type = (
            "categorical" if arg in QUAL_PALETTES
            else variable_type(data, boolean_type="categorical")
        )

        if var_type == "categorical":
            # TODO do this business here or just define with string and resolve later?
            levels = categorical_order(data)
            colors = color_palette(arg, len(levels))
            return Nominal(dict(zip(levels, colors)))

        elif var_type == "numeric":
            return Continuous(arg)

        # TODO just to see when we get here
        assert False

    def get_norm(self, arg, data, transform):

        if isinstance(arg, tuple):
            vmin, vmax = transform(arg)
        else:
            vmin, vmax = transform((data.min(), data.max()))
        norm = mpl.colors.Normalize(vmin, vmax)
        return norm

    def get_mapping(self, arg, data):

        if arg is None:
            # TODO data-dependent default type
            return color_palette("ch:", as_cmap=True)
        elif isinstance(arg, tuple):
            return blend_palette(arg, as_cmap=True)
        elif isinstance(arg, str):
            return color_palette(arg, as_cmap=True)
        assert False
