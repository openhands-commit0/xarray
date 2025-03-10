from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, Union, overload
try:
    import hypothesis.strategies as st
except ImportError as e:
    raise ImportError('`xarray.testing.strategies` requires `hypothesis` to be installed.') from e
import hypothesis.extra.numpy as npst
import numpy as np
from hypothesis.errors import InvalidArgument
import xarray as xr
from xarray.core.types import T_DuckArray
if TYPE_CHECKING:
    from xarray.core.types import _DTypeLikeNested, _ShapeLike
__all__ = ['supported_dtypes', 'pandas_index_dtypes', 'names', 'dimension_names', 'dimension_sizes', 'attrs', 'variables', 'unique_subset_of']

class ArrayStrategyFn(Protocol[T_DuckArray]):

    def __call__(self, *, shape: '_ShapeLike', dtype: '_DTypeLikeNested') -> st.SearchStrategy[T_DuckArray]:
        ...

def supported_dtypes() -> st.SearchStrategy[np.dtype]:
    """
    Generates only those numpy dtypes which xarray can handle.

    Use instead of hypothesis.extra.numpy.scalar_dtypes in order to exclude weirder dtypes such as unicode, byte_string, array, or nested dtypes.
    Also excludes datetimes, which dodges bugs with pandas non-nanosecond datetime overflows.  Checks only native endianness.

    Requires the hypothesis package to be installed.

    See Also
    --------
    :ref:`testing.hypothesis`_
    """
    pass

def pandas_index_dtypes() -> st.SearchStrategy[np.dtype]:
    """
    Dtypes supported by pandas indexes.
    Restrict datetime64 and timedelta64 to ns frequency till Xarray relaxes that.
    """
    pass
_readable_characters = st.characters(categories=['L', 'N'], max_codepoint=383)

def names() -> st.SearchStrategy[str]:
    """
    Generates arbitrary string names for dimensions / variables.

    Requires the hypothesis package to be installed.

    See Also
    --------
    :ref:`testing.hypothesis`_
    """
    pass

def dimension_names(*, name_strategy=names(), min_dims: int=0, max_dims: int=3) -> st.SearchStrategy[list[Hashable]]:
    """
    Generates an arbitrary list of valid dimension names.

    Requires the hypothesis package to be installed.

    Parameters
    ----------
    name_strategy
        Strategy for making names. Useful if we need to share this.
    min_dims
        Minimum number of dimensions in generated list.
    max_dims
        Maximum number of dimensions in generated list.
    """
    pass

def dimension_sizes(*, dim_names: st.SearchStrategy[Hashable]=names(), min_dims: int=0, max_dims: int=3, min_side: int=1, max_side: Union[int, None]=None) -> st.SearchStrategy[Mapping[Hashable, int]]:
    """
    Generates an arbitrary mapping from dimension names to lengths.

    Requires the hypothesis package to be installed.

    Parameters
    ----------
    dim_names: strategy generating strings, optional
        Strategy for generating dimension names.
        Defaults to the `names` strategy.
    min_dims: int, optional
        Minimum number of dimensions in generated list.
        Default is 1.
    max_dims: int, optional
        Maximum number of dimensions in generated list.
        Default is 3.
    min_side: int, optional
        Minimum size of a dimension.
        Default is 1.
    max_side: int, optional
        Minimum size of a dimension.
        Default is `min_length` + 5.

    See Also
    --------
    :ref:`testing.hypothesis`_
    """
    pass
_readable_strings = st.text(_readable_characters, max_size=5)
_attr_keys = _readable_strings
_small_arrays = npst.arrays(shape=npst.array_shapes(max_side=2, max_dims=2), dtype=npst.scalar_dtypes() | npst.byte_string_dtypes() | npst.unicode_string_dtypes())
_attr_values = st.none() | st.booleans() | _readable_strings | _small_arrays
simple_attrs = st.dictionaries(_attr_keys, _attr_values)

def attrs() -> st.SearchStrategy[Mapping[Hashable, Any]]:
    """
    Generates arbitrary valid attributes dictionaries for xarray objects.

    The generated dictionaries can potentially be recursive.

    Requires the hypothesis package to be installed.

    See Also
    --------
    :ref:`testing.hypothesis`_
    """
    pass

@st.composite
def variables(draw: st.DrawFn, *, array_strategy_fn: Union[ArrayStrategyFn, None]=None, dims: Union[st.SearchStrategy[Union[Sequence[Hashable], Mapping[Hashable, int]]], None]=None, dtype: st.SearchStrategy[np.dtype]=supported_dtypes(), attrs: st.SearchStrategy[Mapping]=attrs()) -> xr.Variable:
    """
    Generates arbitrary xarray.Variable objects.

    Follows the basic signature of the xarray.Variable constructor, but allows passing alternative strategies to
    generate either numpy-like array data or dimensions. Also allows specifying the shape or dtype of the wrapped array
    up front.

    Passing nothing will generate a completely arbitrary Variable (containing a numpy array).

    Requires the hypothesis package to be installed.

    Parameters
    ----------
    array_strategy_fn: Callable which returns a strategy generating array-likes, optional
        Callable must only accept shape and dtype kwargs, and must generate results consistent with its input.
        If not passed the default is to generate a small numpy array with one of the supported_dtypes.
    dims: Strategy for generating the dimensions, optional
        Can either be a strategy for generating a sequence of string dimension names,
        or a strategy for generating a mapping of string dimension names to integer lengths along each dimension.
        If provided as a mapping the array shape will be passed to array_strategy_fn.
        Default is to generate arbitrary dimension names for each axis in data.
    dtype: Strategy which generates np.dtype objects, optional
        Will be passed in to array_strategy_fn.
        Default is to generate any scalar dtype using supported_dtypes.
        Be aware that this default set of dtypes includes some not strictly allowed by the array API standard.
    attrs: Strategy which generates dicts, optional
        Default is to generate a nested attributes dictionary containing arbitrary strings, booleans, integers, Nones,
        and numpy arrays.

    Returns
    -------
    variable_strategy
        Strategy for generating xarray.Variable objects.

    Raises
    ------
    ValueError
        If a custom array_strategy_fn returns a strategy which generates an example array inconsistent with the shape
        & dtype input passed to it.

    Examples
    --------
    Generate completely arbitrary Variable objects backed by a numpy array:

    >>> variables().example()  # doctest: +SKIP
    <xarray.Variable (żō: 3)>
    array([43506,   -16,  -151], dtype=int32)
    >>> variables().example()  # doctest: +SKIP
    <xarray.Variable (eD: 4, ğŻżÂĕ: 2, T: 2)>
    array([[[-10000000., -10000000.],
            [-10000000., -10000000.]],
           [[-10000000., -10000000.],
            [        0., -10000000.]],
           [[        0., -10000000.],
            [-10000000.,        inf]],
           [[       -0., -10000000.],
            [-10000000.,        -0.]]], dtype=float32)
    Attributes:
        śřĴ:      {'ĉ': {'iĥf': array([-30117,  -1740], dtype=int16)}}

    Generate only Variable objects with certain dimension names:

    >>> variables(dims=st.just(["a", "b"])).example()  # doctest: +SKIP
    <xarray.Variable (a: 5, b: 3)>
    array([[       248, 4294967295, 4294967295],
           [2412855555, 3514117556, 4294967295],
           [       111, 4294967295, 4294967295],
           [4294967295, 1084434988,      51688],
           [     47714,        252,      11207]], dtype=uint32)

    Generate only Variable objects with certain dimension names and lengths:

    >>> variables(dims=st.just({"a": 2, "b": 1})).example()  # doctest: +SKIP
    <xarray.Variable (a: 2, b: 1)>
    array([[-1.00000000e+007+3.40282347e+038j],
           [-2.75034266e-225+2.22507386e-311j]])

    See Also
    --------
    :ref:`testing.hypothesis`_
    """
    pass

@st.composite
def unique_subset_of(draw: st.DrawFn, objs: Union[Sequence[Hashable], Mapping[Hashable, Any]], *, min_size: int=0, max_size: Union[int, None]=None) -> Union[Sequence[Hashable], Mapping[Hashable, Any]]:
    """
    Return a strategy which generates a unique subset of the given objects.

    Each entry in the output subset will be unique (if input was a sequence) or have a unique key (if it was a mapping).

    Requires the hypothesis package to be installed.

    Parameters
    ----------
    objs: Union[Sequence[Hashable], Mapping[Hashable, Any]]
        Objects from which to sample to produce the subset.
    min_size: int, optional
        Minimum size of the returned subset. Default is 0.
    max_size: int, optional
        Maximum size of the returned subset. Default is the full length of the input.
        If set to 0 the result will be an empty mapping.

    Returns
    -------
    unique_subset_strategy
        Strategy generating subset of the input.

    Examples
    --------
    >>> unique_subset_of({"x": 2, "y": 3}).example()  # doctest: +SKIP
    {'y': 3}
    >>> unique_subset_of(["x", "y"]).example()  # doctest: +SKIP
    ['x']

    See Also
    --------
    :ref:`testing.hypothesis`_
    """
    pass