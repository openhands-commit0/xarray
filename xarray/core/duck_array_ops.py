"""Compatibility module defining operations on duck numpy-arrays.

Currently, this means Dask or NumPy arrays. None of these functions should
accept or return xarray objects.
"""
from __future__ import annotations
import contextlib
import datetime
import inspect
import warnings
from functools import partial
from importlib import import_module
import numpy as np
import pandas as pd
from numpy import all as array_all
from numpy import any as array_any
from numpy import around, full_like, gradient, isclose, isin, isnat, take, tensordot, transpose, unravel_index
from numpy import concatenate as _concatenate
from numpy.lib.stride_tricks import sliding_window_view
from packaging.version import Version
from pandas.api.types import is_extension_array_dtype
from xarray.core import dask_array_ops, dtypes, nputils
from xarray.core.options import OPTIONS
from xarray.core.utils import is_duck_array, is_duck_dask_array, module_available
from xarray.namedarray import pycompat
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import array_type, is_chunked_array
if module_available('numpy', minversion='2.0.0.dev0'):
    from numpy.lib.array_utils import normalize_axis_index
else:
    from numpy.core.multiarray import normalize_axis_index
dask_available = module_available('dask')

def _dask_or_eager_func(name, eager_module=np, dask_module='dask.array'):
    """Create a function that dispatches to dask for dask array inputs."""
    def wrapper(*args, **kwargs):
        if any(is_duck_dask_array(arg) for arg in args):
            try:
                module = import_module(dask_module)
            except ImportError:
                raise ImportError(f"Cannot use {name} with dask arrays without installing dask.")
            func = getattr(module, name)
        else:
            func = getattr(eager_module, name)
        return func(*args, **kwargs)
    return wrapper

def _create_nan_agg_method(name: str, coerce_strings: bool=False, invariant_0d: bool=False):
    """Create a function that dispatches to bottleneck, numbagg, or numpy based on OPTIONS."""
    def method(values, axis=None, skipna=None, **kwargs):
        # For strings, skip bottleneck or numbagg and use the numpy or dask version
        if coerce_strings and values.dtype.kind in {'U', 'S', 'O'}:
            if skipna or (skipna is None and values.dtype.kind == 'O'):
                values = getattr(np, name)(values, axis=axis, **kwargs)
            else:
                values = getattr(np, name)(values, axis=axis, **kwargs)
            return values

        if skipna or (skipna is None and values.dtype.kind not in {'b', 'i', 'u'}):
            if is_duck_dask_array(values):
                module = import_module('dask.array')
                func = getattr(module, f'nan{name}')
                return func(values, axis=axis, **kwargs)
            else:
                func = getattr(nputils, f'nan{name}')
                return func(values, axis=axis, **kwargs)
        else:
            if is_duck_dask_array(values):
                module = import_module('dask.array')
                func = getattr(module, name)
                return func(values, axis=axis, **kwargs)
            else:
                return getattr(np, name)(values, axis=axis, **kwargs)

    method.numeric_only = False
    method.available_min_count = False
    return method
pandas_isnull = _dask_or_eager_func('isnull', eager_module=pd, dask_module='dask.array')
around.__doc__ = str.replace(around.__doc__ or '', 'array([0.,  2.])', 'array([0., 2.])')
around.__doc__ = str.replace(around.__doc__ or '', 'array([0.,  2.])', 'array([0., 2.])')
around.__doc__ = str.replace(around.__doc__ or '', 'array([0.4,  1.6])', 'array([0.4, 1.6])')
around.__doc__ = str.replace(around.__doc__ or '', 'array([0.,  2.,  2.,  4.,  4.])', 'array([0., 2., 2., 4., 4.])')
around.__doc__ = str.replace(around.__doc__ or '', '    .. [2] "How Futile are Mindless Assessments of\n           Roundoff in Floating-Point Computation?", William Kahan,\n           https://people.eecs.berkeley.edu/~wkahan/Mindless.pdf\n', '')
masked_invalid = _dask_or_eager_func('masked_invalid', eager_module=np.ma, dask_module='dask.array.ma')

def as_shared_dtype(scalars_or_arrays, xp=None):
    """Cast a arrays to a shared dtype using xarray's type promotion rules."""
    if not scalars_or_arrays:
        return []
    
    if xp is None:
        xp = np
    
    # Convert all inputs to arrays
    arrays = [xp.asarray(x) for x in scalars_or_arrays]
    
    # Get the target dtype using type promotion rules
    target_dtype = dtypes.result_type(*arrays)
    
    # Cast all arrays to the target dtype
    return [xp.asarray(arr, dtype=target_dtype) for arr in arrays]

def lazy_array_equiv(arr1, arr2):
    """Like array_equal, but doesn't actually compare values.
    Returns True when arr1, arr2 identical or their dask tokens are equal.
    Returns False when shapes are not equal.
    Returns None when equality cannot determined: one or both of arr1, arr2 are numpy arrays;
    or their dask tokens are not equal
    """
    if arr1 is arr2:
        return True
    
    if arr1 is None or arr2 is None:
        return arr1 is None and arr2 is None
    
    if not is_duck_array(arr1) or not is_duck_array(arr2):
        return None
    
    if arr1.shape != arr2.shape:
        return False
    
    if is_duck_dask_array(arr1) and is_duck_dask_array(arr2):
        from dask.base import tokenize
        return tokenize(arr1) == tokenize(arr2)
    
    return None

def allclose_or_equiv(arr1, arr2, rtol=1e-05, atol=1e-08):
    """Like np.allclose, but also allows values to be NaN in both arrays"""
    arr1, arr2 = as_shared_dtype([arr1, arr2])
    
    if arr1.shape != arr2.shape:
        return False
    
    if is_duck_dask_array(arr1) or is_duck_dask_array(arr2):
        import dask.array as da
        arr1 = da.asarray(arr1)
        arr2 = da.asarray(arr2)
        
        # Check if arrays are equal (including NaN)
        equal_nan = da.isnan(arr1) & da.isnan(arr2)
        close = da.isclose(arr1, arr2, rtol=rtol, atol=atol, equal_nan=True)
        return bool(da.all(equal_nan | close).compute())
    
    # Check if arrays are equal (including NaN)
    equal_nan = np.isnan(arr1) & np.isnan(arr2)
    close = np.isclose(arr1, arr2, rtol=rtol, atol=atol, equal_nan=True)
    return bool(np.all(equal_nan | close))

def array_equiv(arr1, arr2):
    """Like np.array_equal, but also allows values to be NaN in both arrays"""
    arr1, arr2 = as_shared_dtype([arr1, arr2])
    
    if arr1.shape != arr2.shape:
        return False
    
    if is_duck_dask_array(arr1) or is_duck_dask_array(arr2):
        import dask.array as da
        arr1 = da.asarray(arr1)
        arr2 = da.asarray(arr2)
        
        # Check if arrays are equal (including NaN)
        equal_nan = da.isnan(arr1) & da.isnan(arr2)
        equal = arr1 == arr2
        return bool(da.all(equal_nan | equal).compute())
    
    # Check if arrays are equal (including NaN)
    equal_nan = np.isnan(arr1) & np.isnan(arr2)
    equal = arr1 == arr2
    return bool(np.all(equal_nan | equal))

def array_notnull_equiv(arr1, arr2):
    """Like np.array_equal, but also allows values to be NaN in either or both
    arrays
    """
    arr1, arr2 = as_shared_dtype([arr1, arr2])
    
    if arr1.shape != arr2.shape:
        return False
    
    if is_duck_dask_array(arr1) or is_duck_dask_array(arr2):
        import dask.array as da
        arr1 = da.asarray(arr1)
        arr2 = da.asarray(arr2)
        
        # Check if arrays are equal where neither is NaN
        valid = ~(da.isnan(arr1) | da.isnan(arr2))
        equal = arr1 == arr2
        return bool(da.all(~valid | equal).compute())
    
    # Check if arrays are equal where neither is NaN
    valid = ~(np.isnan(arr1) | np.isnan(arr2))
    equal = arr1 == arr2
    return bool(np.all(~valid | equal))

def count(data, axis=None):
    """Count the number of non-NA in this array along the given axis or axes"""
    if is_duck_dask_array(data):
        import dask.array as da
        return da.count_nonzero(~da.isnan(data), axis=axis)
    else:
        return np.count_nonzero(~np.isnan(data), axis=axis)

def where(condition, x, y):
    """Three argument where() with better dtype promotion rules."""
    if is_duck_dask_array(condition) or is_duck_dask_array(x) or is_duck_dask_array(y):
        import dask.array as da
        return da.where(condition, x, y)
    else:
        x, y = as_shared_dtype([x, y])
        return np.where(condition, x, y)

def concatenate(arrays, axis=0):
    """concatenate() with better dtype promotion rules."""
    if not arrays:
        return np.array([], dtype=object)
    
    if any(is_duck_dask_array(arr) for arr in arrays):
        import dask.array as da
        arrays = [da.asarray(arr) for arr in arrays]
        return da.concatenate(arrays, axis=axis)
    else:
        arrays = as_shared_dtype(arrays)
        return _concatenate(arrays, axis=axis)

def stack(arrays, axis=0):
    """stack() with better dtype promotion rules."""
    if not arrays:
        return np.array([], dtype=object)
    
    if any(is_duck_dask_array(arr) for arr in arrays):
        import dask.array as da
        arrays = [da.asarray(arr) for arr in arrays]
        return da.stack(arrays, axis=axis)
    else:
        arrays = as_shared_dtype(arrays)
        return np.stack(arrays, axis=axis)
argmax = _create_nan_agg_method('argmax', coerce_strings=True)
argmin = _create_nan_agg_method('argmin', coerce_strings=True)
max = _create_nan_agg_method('max', coerce_strings=True, invariant_0d=True)
min = _create_nan_agg_method('min', coerce_strings=True, invariant_0d=True)
sum = _create_nan_agg_method('sum', invariant_0d=True)
sum.numeric_only = True
sum.available_min_count = True
std = _create_nan_agg_method('std')
std.numeric_only = True
var = _create_nan_agg_method('var')
var.numeric_only = True
median = _create_nan_agg_method('median', invariant_0d=True)
median.numeric_only = True
prod = _create_nan_agg_method('prod', invariant_0d=True)
prod.numeric_only = True
prod.available_min_count = True
cumprod_1d = _create_nan_agg_method('cumprod', invariant_0d=True)
cumprod_1d.numeric_only = True
cumsum_1d = _create_nan_agg_method('cumsum', invariant_0d=True)
cumsum_1d.numeric_only = True
_mean = _create_nan_agg_method('mean', invariant_0d=True)

def _datetime_nanmin(array):
    """nanmin() function for datetime64.

    Caveats that this function deals with:

    - In numpy < 1.18, min() on datetime64 incorrectly ignores NaT
    - numpy nanmin() don't work on datetime64 (all versions at the moment of writing)
    - dask min() does not work on datetime64 (all versions at the moment of writing)
    """
    if is_duck_dask_array(array):
        import dask.array as da
        mask = ~da.isnat(array)
        valid = array[mask]
        if valid.size == 0:
            return np.datetime64('NaT')
        return da.min(valid).compute()
    else:
        mask = ~isnat(array)
        valid = array[mask]
        if valid.size == 0:
            return np.datetime64('NaT')
        return valid.min()

def datetime_to_numeric(array, offset=None, datetime_unit=None, dtype=float):
    """Convert an array containing datetime-like data to numerical values.
    Convert the datetime array to a timedelta relative to an offset.
    Parameters
    ----------
    array : array-like
        Input data
    offset : None, datetime or cftime.datetime
        Datetime offset. If None, this is set by default to the array's minimum
        value to reduce round off errors.
    datetime_unit : {None, Y, M, W, D, h, m, s, ms, us, ns, ps, fs, as}
        If not None, convert output to a given datetime unit. Note that some
        conversions are not allowed due to non-linear relationships between units.
    dtype : dtype
        Output dtype.
    Returns
    -------
    array
        Numerical representation of datetime object relative to an offset.
    Notes
    -----
    Some datetime unit conversions won't work, for example from days to years, even
    though some calendars would allow for them (e.g. no_leap). This is because there
    is no `cftime.timedelta` object.
    """
    if offset is None:
        offset = _datetime_nanmin(array)
    
    if is_duck_dask_array(array):
        import dask.array as da
        result = da.subtract(array, offset)
    else:
        result = array - offset
    
    if datetime_unit is not None:
        if isinstance(result, np.ndarray):
            result = np_timedelta64_to_float(result, datetime_unit)
        else:
            result = timedelta_to_numeric(result, datetime_unit)
    
    return result.astype(dtype)

def timedelta_to_numeric(value, datetime_unit='ns', dtype=float):
    """Convert a timedelta-like object to numerical values.

    Parameters
    ----------
    value : datetime.timedelta, numpy.timedelta64, pandas.Timedelta, str
        Time delta representation.
    datetime_unit : {Y, M, W, D, h, m, s, ms, us, ns, ps, fs, as}
        The time units of the output values. Note that some conversions are not allowed due to
        non-linear relationships between units.
    dtype : type
        The output data type.

    """
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.timedelta64):
            return np_timedelta64_to_float(value, datetime_unit)
        else:
            return value.astype(dtype)
    elif isinstance(value, pd.Timedelta):
        return pd_timedelta_to_float(value, datetime_unit)
    elif isinstance(value, timedelta):
        return py_timedelta_to_float(value, datetime_unit)
    else:
        return value

def np_timedelta64_to_float(array, datetime_unit):
    """Convert numpy.timedelta64 to float.

    Notes
    -----
    The array is first converted to microseconds, which is less likely to
    cause overflow errors.
    """
    # Convert to microseconds first to avoid overflow errors
    microseconds = array.astype('timedelta64[us]').astype(float)
    
    # Convert from microseconds to target unit
    if datetime_unit == 'us':
        return microseconds
    elif datetime_unit == 'ns':
        return microseconds * 1000
    elif datetime_unit == 'ms':
        return microseconds / 1000
    elif datetime_unit == 's':
        return microseconds / 1_000_000
    elif datetime_unit == 'm':
        return microseconds / (60 * 1_000_000)
    elif datetime_unit == 'h':
        return microseconds / (3600 * 1_000_000)
    elif datetime_unit == 'D':
        return microseconds / (86400 * 1_000_000)
    else:
        raise ValueError(f"Unsupported datetime unit: {datetime_unit}")

def pd_timedelta_to_float(value, datetime_unit):
    """Convert pandas.Timedelta to float.

    Notes
    -----
    Built on the assumption that pandas timedelta values are in nanoseconds,
    which is also the numpy default resolution.
    """
    # Convert to nanoseconds first
    nanoseconds = float(value.value)
    
    # Convert from nanoseconds to target unit
    if datetime_unit == 'ns':
        return nanoseconds
    elif datetime_unit == 'us':
        return nanoseconds / 1000
    elif datetime_unit == 'ms':
        return nanoseconds / 1_000_000
    elif datetime_unit == 's':
        return nanoseconds / 1_000_000_000
    elif datetime_unit == 'm':
        return nanoseconds / (60 * 1_000_000_000)
    elif datetime_unit == 'h':
        return nanoseconds / (3600 * 1_000_000_000)
    elif datetime_unit == 'D':
        return nanoseconds / (86400 * 1_000_000_000)
    else:
        raise ValueError(f"Unsupported datetime unit: {datetime_unit}")

def py_timedelta_to_float(array, datetime_unit):
    """Convert a timedelta object to a float, possibly at a loss of resolution."""
    # Convert to total seconds first
    total_seconds = array.total_seconds()
    
    # Convert from seconds to target unit
    if datetime_unit == 's':
        return total_seconds
    elif datetime_unit == 'ns':
        return total_seconds * 1_000_000_000
    elif datetime_unit == 'us':
        return total_seconds * 1_000_000
    elif datetime_unit == 'ms':
        return total_seconds * 1000
    elif datetime_unit == 'm':
        return total_seconds / 60
    elif datetime_unit == 'h':
        return total_seconds / 3600
    elif datetime_unit == 'D':
        return total_seconds / 86400
    else:
        raise ValueError(f"Unsupported datetime unit: {datetime_unit}")

def mean(array, axis=None, skipna=None, **kwargs):
    """inhouse mean that can handle np.datetime64 or cftime.datetime
    dtypes"""
    if array.dtype.kind in {'M', 'm'}:
        offset = _datetime_nanmin(array)
        numeric_array = datetime_to_numeric(array, offset)
        numeric_mean = _mean(numeric_array, axis=axis, skipna=skipna, **kwargs)
        
        if is_duck_dask_array(array):
            import dask.array as da
            result = da.add(offset, numeric_mean)
        else:
            result = offset + numeric_mean
        return result
    else:
        return _mean(array, axis=axis, skipna=skipna, **kwargs)
mean.numeric_only = True

def cumprod(array, axis=None, **kwargs):
    """N-dimensional version of cumprod."""
    if axis is None:
        array = array.ravel()
        axis = 0
    
    if is_duck_dask_array(array):
        import dask.array as da
        return da.cumprod(array, axis=axis, **kwargs)
    else:
        return cumprod_1d(array, axis=axis, **kwargs)

def cumsum(array, axis=None, **kwargs):
    """N-dimensional version of cumsum."""
    if axis is None:
        array = array.ravel()
        axis = 0
    
    if is_duck_dask_array(array):
        import dask.array as da
        return da.cumsum(array, axis=axis, **kwargs)
    else:
        return cumsum_1d(array, axis=axis, **kwargs)

def first(values, axis, skipna=None):
    """Return the first non-NA elements in this array along the given axis"""
    if skipna or (skipna is None and values.dtype.kind not in {'b', 'i', 'u'}):
        mask = ~pandas_isnull(values)
        if mask.all():
            # Return the first value since there are no NAs
            return take(values, 0, axis=axis)
        else:
            # Find the first valid value and return it
            first_index = argmax(mask, axis=axis)
            return take_from_dim_n(values, first_index, axis)
    else:
        # No skipping NA - just return the first value
        return take(values, 0, axis=axis)

def last(values, axis, skipna=None):
    """Return the last non-NA elements in this array along the given axis"""
    if skipna or (skipna is None and values.dtype.kind not in {'b', 'i', 'u'}):
        mask = ~pandas_isnull(values)
        if mask.all():
            # Return the last value since there are no NAs
            return take(values, -1, axis=axis)
        else:
            # Find the last valid value and return it
            last_index = values.shape[axis] - 1 - argmax(mask[::-1], axis=axis)
            return take_from_dim_n(values, last_index, axis)
    else:
        # No skipping NA - just return the last value
        return take(values, -1, axis=axis)

def least_squares(lhs, rhs, rcond=None, skipna=False):
    """Return the coefficients and residuals of a least-squares fit."""
    if skipna:
        # Mask out NaN values
        mask = ~(pandas_isnull(lhs) | pandas_isnull(rhs))
        if not mask.all():
            lhs = lhs[mask]
            rhs = rhs[mask]
    
    if is_duck_dask_array(lhs) or is_duck_dask_array(rhs):
        import dask.array as da
        lhs = da.asarray(lhs)
        rhs = da.asarray(rhs)
        return da.linalg.lstsq(lhs, rhs, rcond=rcond)
    else:
        return np.linalg.lstsq(lhs, rhs, rcond=rcond)

def astype(data, dtype, copy=True):
    """Cast data array to dtype, properly handling dask arrays."""
    if is_duck_dask_array(data):
        import dask.array as da
        return da.astype(data, dtype=dtype, copy=copy)
    else:
        return np.asarray(data, dtype=dtype)

def _push(array, n: int | None=None, axis: int=-1):
    """
    Use either bottleneck or numbagg depending on options & what's available
    """
    if is_duck_dask_array(array):
        import dask.array as da
        return da.roll(array, shift=n or 1, axis=axis)
    else:
        if OPTIONS["use_numbagg"] and module_available("numbagg"):
            import numbagg
            return numbagg.push(array, n=n, axis=axis)
        elif OPTIONS["use_bottleneck"] and _BOTTLENECK_AVAILABLE:
            return bn.push(array, n=n, axis=axis)
        else:
            return np.roll(array, shift=n or 1, axis=axis)