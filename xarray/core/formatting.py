"""String formatting routines for __repr__.
"""
from __future__ import annotations
import contextlib
import functools
import math
from collections import defaultdict
from collections.abc import Collection, Hashable, Sequence
from datetime import datetime, timedelta
from itertools import chain, zip_longest
from reprlib import recursive_repr
from textwrap import dedent
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from pandas.errors import OutOfBoundsDatetime
from xarray.core.datatree_render import RenderDataTree
from xarray.core.duck_array_ops import array_equiv, astype
from xarray.core.indexing import MemoryCachedArray
from xarray.core.iterators import LevelOrderIter
from xarray.core.options import OPTIONS, _get_boolean_with_default
from xarray.core.utils import is_duck_array
from xarray.namedarray.pycompat import array_type, to_duck_array, to_numpy
if TYPE_CHECKING:
    from xarray.core.coordinates import AbstractCoordinates
    from xarray.core.datatree import DataTree
UNITS = ('B', 'kB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB')

def _mapping_repr(mapping, title, summarizer, expand_option_name=None, col_width=None):
    """Create a one-line summary of a mapping object."""
    if not mapping:
        return f"{title}:\n{EMPTY_REPR}"

    if expand_option_name is not None and _get_boolean_with_default(expand_option_name):
        summary = []
        for k, v in mapping.items():
            summary.append(summarizer(k, v, col_width=col_width))
    else:
        summary = [summarizer(k, v, col_width=col_width) for k, v in mapping.items()]

    return f"{title}:\n" + "\n".join(summary)

def pretty_print(x, numchars: int):
    """Given an object `x`, call `str(x)` and format the returned string so
    that it is numchars long, padding with trailing spaces or truncating with
    ellipses as necessary
    """
    s = str(x)
    if len(s) > numchars:
        return s[:(numchars - 3)] + "..."
    else:
        return s.ljust(numchars)

def first_n_items(array, n_desired):
    """Returns the first n_desired items of an array"""
    if array is None:
        return []
    return array[0:n_desired]

def last_n_items(array, n_desired):
    """Returns the last n_desired items of an array"""
    if array is None:
        return []
    return array[-n_desired:]

def last_item(array):
    """Returns the last item of an array in a list or an empty list."""
    if array is None or len(array) == 0:
        return []
    return [array[-1]]

def calc_max_rows_first(max_rows: int) -> int:
    """Calculate the first rows to maintain the max number of rows."""
    if max_rows is None:
        return None
    return max(1, (max_rows + 1) // 2)

def calc_max_rows_last(max_rows: int) -> int:
    """Calculate the last rows to maintain the max number of rows."""
    if max_rows is None:
        return None
    return max(1, max_rows // 2)

def format_timestamp(t):
    """Cast given object to a Timestamp and return a nicely formatted string"""
    try:
        datetime_str = pd.Timestamp(t).isoformat()
        try:
            date_str, time_str = datetime_str.split('T')
        except ValueError:
            # catch NaT and others that don't split nicely
            return datetime_str
        else:
            if time_str == '00:00:00':
                return date_str
            else:
                return f'{date_str} {time_str}'
    except OutOfBoundsDatetime:
        return str(t)

def format_timedelta(t, timedelta_format=None):
    """Cast given object to a Timestamp and return a nicely formatted string"""
    if timedelta_format is None:
        timedelta_format = 'auto'

    if isinstance(t, pd.Timedelta):
        t = t.to_pytimedelta()

    if timedelta_format == 'date':
        return str(t)
    elif timedelta_format == 'auto':
        if t == pd.Timedelta(0):
            return '0:00:00'
        else:
            total_seconds = t.total_seconds()
            days = int(total_seconds // (24 * 3600))
            remainder = total_seconds % (24 * 3600)
            hours = int(remainder // 3600)
            remainder = remainder % 3600
            minutes = int(remainder // 60)
            seconds = int(remainder % 60)
            microseconds = int(t.microseconds)

            if days == 0:
                if microseconds:
                    return f'{hours:02d}:{minutes:02d}:{seconds:02d}.{microseconds:06d}'
                else:
                    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'
            else:
                return f'{days} days {hours:02d}:{minutes:02d}:{seconds:02d}'
    else:
        raise ValueError(f"Unknown timedelta_format: {timedelta_format}")

def format_item(x, timedelta_format=None, quote_strings=True):
    """Returns a succinct summary of an object as a string"""
    if isinstance(x, (np.datetime64, datetime)):
        return format_timestamp(x)
    elif isinstance(x, (np.timedelta64, timedelta, pd.Timedelta)):
        return format_timedelta(x, timedelta_format=timedelta_format)
    elif isinstance(x, (str, bytes)):
        return repr(x) if quote_strings else str(x)
    elif isinstance(x, (float, np.float_)):
        return f'{x:.4g}'
    else:
        return str(x)

def format_items(x):
    """Returns a succinct summaries of all items in a sequence as strings"""
    if is_duck_array(x):
        x = to_numpy(x)
    return [format_item(xi) for xi in x]

def format_array_flat(array, max_width: int):
    """Return a formatted string for as many items in the flattened version of
    array that will fit within max_width characters.
    """
    pass
_KNOWN_TYPE_REPRS = {('numpy', 'ndarray'): 'np.ndarray', ('sparse._coo.core', 'COO'): 'sparse.COO'}

def inline_dask_repr(array):
    """Similar to dask.array.DataArray.__repr__, but without
    redundant information that's already printed by the repr
    function of the xarray wrapper.
    """
    pass

def inline_sparse_repr(array):
    """Similar to sparse.COO.__repr__, but without the redundant shape/dtype."""
    pass

def inline_variable_array_repr(var, max_width):
    """Build a one-line summary of a variable's data."""
    pass

def summarize_variable(name: Hashable, var, col_width: int, max_width: int | None=None, is_index: bool=False):
    """Summarize a variable in one line, e.g., for the Dataset.__repr__."""
    pass

def summarize_attr(key, value, col_width=None):
    """Summary for __repr__ - use ``X.attrs[key]`` for full value."""
    pass
EMPTY_REPR = '    *empty*'
data_vars_repr = functools.partial(_mapping_repr, title='Data variables', summarizer=summarize_variable, expand_option_name='display_expand_data_vars')
attrs_repr = functools.partial(_mapping_repr, title='Attributes', summarizer=summarize_attr, expand_option_name='display_expand_attrs')

def _element_formatter(elements: Collection[Hashable], col_width: int, max_rows: int | None=None, delimiter: str=', ') -> str:
    """
    Formats elements for better readability.

    Once it becomes wider than the display width it will create a newline and
    continue indented to col_width.
    Once there are more rows than the maximum displayed rows it will start
    removing rows.

    Parameters
    ----------
    elements : Collection of hashable
        Elements to join together.
    col_width : int
        The width to indent to if a newline has been made.
    max_rows : int, optional
        The maximum number of allowed rows. The default is None.
    delimiter : str, optional
        Delimiter to use between each element. The default is ", ".
    """
    pass

def limit_lines(string: str, *, limit: int):
    """
    If the string is more lines than the limit,
    this returns the middle lines replaced by an ellipsis
    """
    pass

def short_data_repr(array):
    """Format "data" for DataArray and Variable."""
    pass

def dims_and_coords_repr(ds) -> str:
    """Partial Dataset repr for use inside DataTree inheritance errors."""
    pass
diff_data_vars_repr = functools.partial(_diff_mapping_repr, title='Data variables', summarizer=summarize_variable)
diff_attrs_repr = functools.partial(_diff_mapping_repr, title='Attributes', summarizer=summarize_attr)

def diff_treestructure(a: DataTree, b: DataTree, require_names_equal: bool) -> str:
    """
    Return a summary of why two trees are not isomorphic.
    If they are isomorphic return an empty string.
    """
    pass

def diff_nodewise_summary(a: DataTree, b: DataTree, compat):
    """Iterates over all corresponding nodes, recording differences between data at each location."""
    pass

def _single_node_repr(node: DataTree) -> str:
    """Information about this node, not including its relationships to other nodes."""
    pass

def datatree_repr(dt: DataTree):
    """A printable representation of the structure of this entire tree."""
    pass

def render_human_readable_nbytes(nbytes: int, /, *, attempt_constant_width: bool=False) -> str:
    """Renders simple human-readable byte count representation

    This is only a quick representation that should not be relied upon for precise needs.

    To get the exact byte count, please use the ``nbytes`` attribute directly.

    Parameters
    ----------
    nbytes
        Byte count
    attempt_constant_width
        For reasonable nbytes sizes, tries to render a fixed-width representation.

    Returns
    -------
        Human-readable representation of the byte count
    """
    pass