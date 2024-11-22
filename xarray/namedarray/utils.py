from __future__ import annotations
import importlib
import sys
import warnings
from collections.abc import Hashable, Iterable, Iterator, Mapping
from functools import lru_cache
from typing import TYPE_CHECKING, Any, TypeVar, cast
import numpy as np
from packaging.version import Version
from xarray.namedarray._typing import ErrorOptionsWithWarn, _DimsLike
if TYPE_CHECKING:
    if sys.version_info >= (3, 10):
        from typing import TypeGuard
    else:
        from typing_extensions import TypeGuard
    from numpy.typing import NDArray
    try:
        from dask.array.core import Array as DaskArray
        from dask.typing import DaskCollection
    except ImportError:
        DaskArray = NDArray
        DaskCollection: Any = NDArray
    from xarray.namedarray._typing import _Dim, duckarray
K = TypeVar('K')
V = TypeVar('V')
T = TypeVar('T')

@lru_cache
def module_available(module: str, minversion: str | None=None) -> bool:
    """Checks whether a module is installed without importing it.

    Use this for a lightweight check and lazy imports.

    Parameters
    ----------
    module : str
        Name of the module.
    minversion : str, optional
        Minimum version of the module

    Returns
    -------
    available : bool
        Whether the module is installed.
    """
    try:
        mod = importlib.import_module(module)
    except ImportError:
        return False
    
    if minversion is not None:
        try:
            version = Version(mod.__version__)
        except (AttributeError, ValueError):
            return False
        if version < Version(minversion):
            return False
    
    return True

def is_dict_like(value: Any) -> TypeGuard[Mapping]:
    """Check if a value behaves like a mapping.

    Parameters
    ----------
    value : Any
        Value to check.

    Returns
    -------
    bool
        True if the value behaves like a mapping.
    """
    return hasattr(value, 'keys') and hasattr(value, '__getitem__')

def to_0d_object_array(value: object) -> NDArray[np.object_]:
    """Given a value, wrap it in a 0-D numpy.ndarray with dtype=object."""
    result = np.empty((), dtype=object)
    result[()] = value
    return result

def drop_missing_dims(supplied_dims: Iterable[_Dim], dims: Iterable[_Dim], missing_dims: ErrorOptionsWithWarn) -> _DimsLike:
    """Depending on the setting of missing_dims, drop any dimensions from supplied_dims that
    are not present in dims.

    Parameters
    ----------
    supplied_dims : Iterable of Hashable
    dims : Iterable of Hashable
    missing_dims : {"raise", "warn", "ignore"}
    """
    dims_set = set(dims)
    supplied_dims_list = list(supplied_dims)
    missing = set(supplied_dims_list) - dims_set

    if missing:
        if missing_dims == "raise":
            raise ValueError(f"Dimensions {missing} not found in dims {dims_set}")
        elif missing_dims == "warn":
            warnings.warn(f"Dimensions {missing} not found in dims {dims_set}")
        
        # Filter out missing dims
        supplied_dims_list = [d for d in supplied_dims_list if d in dims_set]

    return supplied_dims_list

def infix_dims(dims_supplied: Iterable[_Dim], dims_all: Iterable[_Dim], missing_dims: ErrorOptionsWithWarn='raise') -> Iterator[_Dim]:
    """
    Resolves a supplied list containing an ellipsis representing other items, to
    a generator with the 'realized' list of all items
    """
    dims_supplied_list = list(dims_supplied)
    dims_all_list = list(dims_all)

    if Ellipsis not in dims_supplied_list:
        for dim in drop_missing_dims(dims_supplied_list, dims_all_list, missing_dims):
            yield dim
        return

    # Find the position of Ellipsis
    ellipsis_pos = dims_supplied_list.index(Ellipsis)
    
    # Get the dims before and after Ellipsis
    before_ellipsis = dims_supplied_list[:ellipsis_pos]
    after_ellipsis = dims_supplied_list[ellipsis_pos + 1:]
    
    # Get the dims that should replace Ellipsis
    dims_all_set = set(dims_all_list)
    dims_specified = set(before_ellipsis + after_ellipsis)
    ellipsis_dims = [d for d in dims_all_list if d not in dims_specified]
    
    # Yield all dims in order
    for dim in before_ellipsis:
        if dim in dims_all_set:
            yield dim
    for dim in ellipsis_dims:
        yield dim
    for dim in after_ellipsis:
        if dim in dims_all_set:
            yield dim

class ReprObject:
    """Object that prints as the given value, for use with sentinel values."""
    __slots__ = ('_value',)
    _value: str

    def __init__(self, value: str):
        self._value = value

    def __repr__(self) -> str:
        return self._value

    def __eq__(self, other: ReprObject | Any) -> bool:
        return self._value == other._value if isinstance(other, ReprObject) else False

    def __hash__(self) -> int:
        return hash((type(self), self._value))

    def __dask_tokenize__(self) -> object:
        from dask.base import normalize_token
        return normalize_token((type(self), self._value))

def is_dask_collection(x: Any) -> TypeGuard[DaskCollection]:
    """Test if an object is a dask collection."""
    try:
        from dask.base import is_dask_collection as _is_dask_collection
        return _is_dask_collection(x)
    except ImportError:
        return False

def is_duck_array(value: Any) -> TypeGuard[duckarray]:
    """Check if value is a duck array."""
    return hasattr(value, '__array_function__') or hasattr(value, '__array_namespace__')

def is_duck_dask_array(value: Any) -> TypeGuard[DaskArray]:
    """Check if value is a dask array."""
    return is_duck_array(value) and is_dask_collection(value)

def either_dict_or_kwargs(pos_kwargs: Mapping[K, V] | None, kw_kwargs: Mapping[str, V], func_name: str | None=None) -> dict[Hashable, V]:
    """Return a single dictionary combining dict and kwargs.

    If both are provided, the values in kw_kwargs take precedence.

    Parameters
    ----------
    pos_kwargs : mapping, optional
        A mapping object to be combined with kw_kwargs.
    kw_kwargs : mapping
        A mapping object to be combined with pos_kwargs.
    func_name : str, optional
        The name of the function being called. This is used to provide a more
        informative error message in case of duplicated keys.

    Returns
    -------
    dict
        A dictionary combining the values of pos_kwargs and kw_kwargs.
    """
    if pos_kwargs is None:
        return dict(kw_kwargs)
    
    if not is_dict_like(pos_kwargs):
        raise ValueError("the first argument must be a dictionary")
    
    combined = dict(pos_kwargs)
    for k, v in kw_kwargs.items():
        if k in combined:
            if func_name is None:
                msg = f"argument {k!r} specified both by position and keyword"
            else:
                msg = f"{func_name}() got multiple values for argument {k!r}"
            raise TypeError(msg)
        combined[k] = v
    
    return combined