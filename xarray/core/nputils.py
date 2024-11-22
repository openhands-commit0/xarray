from __future__ import annotations
import warnings
from typing import Callable
import numpy as np
import pandas as pd
from packaging.version import Version
from xarray.core.utils import is_duck_array, module_available
from xarray.namedarray import pycompat
if module_available('numpy', minversion='2.0.0.dev0'):
    from numpy.lib.array_utils import normalize_axis_index
else:
    from numpy.core.multiarray import normalize_axis_index
try:
    from numpy.exceptions import RankWarning
except ImportError:
    from numpy import RankWarning
from xarray.core.options import OPTIONS
try:
    import bottleneck as bn
    _BOTTLENECK_AVAILABLE = True
except ImportError:
    bn = np
    _BOTTLENECK_AVAILABLE = False

def inverse_permutation(indices: np.ndarray, N: int | None=None) -> np.ndarray:
    """Return indices for an inverse permutation.

    Parameters
    ----------
    indices : 1D np.ndarray with dtype=int
        Integer positions to assign elements to.
    N : int, optional
        Size of the array

    Returns
    -------
    inverse_permutation : 1D np.ndarray with dtype=int
        Integer indices to take from the original array to create the
        permutation.
    """
    if N is None:
        N = len(indices)
    result = np.arange(N)
    result[indices] = np.arange(len(indices))
    return result

def _is_contiguous(positions):
    """Given a non-empty list, does it consist of contiguous integers?"""
    if not positions:
        return True
    return np.array_equal(positions, range(min(positions), max(positions) + 1))

def _advanced_indexer_subspaces(key):
    """Indices of the advanced indexes subspaces for mixed indexing and vindex."""
    if not isinstance(key, tuple):
        key = (key,)
    
    positions = []
    for i, k in enumerate(key):
        if isinstance(k, (np.ndarray, list)) and isinstance(k[0], (int, np.integer)):
            positions.append(i)
    
    mixed_positions = []
    vindex_positions = []
    
    if positions:
        # Split positions into mixed and vindex depending on whether they form
        # contiguous spans when sorted
        spans = []
        start = positions[0]
        end = positions[0]
        for p in positions[1:]:
            if p == end + 1:
                end = p
            else:
                spans.append((start, end))
                start = p
                end = p
        spans.append((start, end))
        
        for start, end in spans:
            if start == end:
                vindex_positions.append(start)
            else:
                mixed_positions.extend(range(start, end + 1))
    
    return mixed_positions, vindex_positions

class NumpyVIndexAdapter:
    """Object that implements indexing like vindex on a np.ndarray.

    This is a pure Python implementation of (some of) the logic in this NumPy
    proposal: https://github.com/numpy/numpy/pull/6256
    """

    def __init__(self, array):
        self._array = array

    def __getitem__(self, key):
        mixed_positions, vindex_positions = _advanced_indexer_subspaces(key)
        return np.moveaxis(self._array[key], mixed_positions, vindex_positions)

    def __setitem__(self, key, value):
        """Value must have dimensionality matching the key."""
        mixed_positions, vindex_positions = _advanced_indexer_subspaces(key)
        self._array[key] = np.moveaxis(value, vindex_positions, mixed_positions)

def _create_method(name: str) -> Callable:
    """Create a method that dispatches to bottleneck or numpy based on OPTIONS."""
    def method(values, axis=None, **kwargs):
        if (
            OPTIONS["use_bottleneck"]
            and _BOTTLENECK_AVAILABLE
            and not isinstance(values, np.ma.MaskedArray)
            and not is_duck_array(values)
            and not kwargs
        ):
            try:
                return getattr(bn, name)(values, axis=axis)
            except (ValueError, AttributeError):
                pass
        return getattr(np, name)(values, axis=axis, **kwargs)
    return method
nanmin = _create_method('nanmin')
nanmax = _create_method('nanmax')
nanmean = _create_method('nanmean')
nanmedian = _create_method('nanmedian')
nanvar = _create_method('nanvar')
nanstd = _create_method('nanstd')
nanprod = _create_method('nanprod')
nancumsum = _create_method('nancumsum')
nancumprod = _create_method('nancumprod')
nanargmin = _create_method('nanargmin')
nanargmax = _create_method('nanargmax')
nanquantile = _create_method('nanquantile')