from __future__ import annotations
from collections.abc import Mapping
from typing import Any, Generic
import numpy as np
from packaging.version import Version
from xarray.core.computation import apply_ufunc
from xarray.core.options import _get_keep_attrs
from xarray.core.pdcompat import count_not_none
from xarray.core.types import T_DataWithCoords
from xarray.core.utils import module_available
from xarray.namedarray import pycompat

def _get_alpha(com: float | None=None, span: float | None=None, halflife: float | None=None, alpha: float | None=None) -> float:
    """
    Convert com, span, halflife to alpha.
    """
    pass

class RollingExp(Generic[T_DataWithCoords]):
    """
    Exponentially-weighted moving window object.
    Similar to EWM in pandas

    Parameters
    ----------
    obj : Dataset or DataArray
        Object to window.
    windows : mapping of hashable to int (or float for alpha type)
        A mapping from the name of the dimension to create the rolling
        exponential window along (e.g. `time`) to the size of the moving window.
    window_type : {"span", "com", "halflife", "alpha"}, default: "span"
        The format of the previously supplied window. Each is a simple
        numerical transformation of the others. Described in detail:
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html

    Returns
    -------
    RollingExp : type of input argument
    """

    def __init__(self, obj: T_DataWithCoords, windows: Mapping[Any, int | float], window_type: str='span', min_weight: float=0.0):
        if not module_available('numbagg'):
            raise ImportError('numbagg >= 0.2.1 is required for rolling_exp but currently numbagg is not installed')
        elif pycompat.mod_version('numbagg') < Version('0.2.1'):
            raise ImportError(f'numbagg >= 0.2.1 is required for rolling_exp but currently version {pycompat.mod_version('numbagg')} is installed')
        elif pycompat.mod_version('numbagg') < Version('0.3.1') and min_weight > 0:
            raise ImportError(f'numbagg >= 0.3.1 is required for `min_weight > 0` within `.rolling_exp` but currently version {pycompat.mod_version('numbagg')} is installed')
        self.obj: T_DataWithCoords = obj
        dim, window = next(iter(windows.items()))
        self.dim = dim
        self.alpha = _get_alpha(**{window_type: window})
        self.min_weight = min_weight
        kwargs = dict(alpha=self.alpha, axis=-1)
        if min_weight > 0:
            kwargs['min_weight'] = min_weight
        self.kwargs = kwargs

    def mean(self, keep_attrs: bool | None=None) -> T_DataWithCoords:
        """
        Exponentially weighted moving average.

        Parameters
        ----------
        keep_attrs : bool, default: None
            If True, the attributes (``attrs``) will be copied from the original
            object to the new one. If False, the new object will be returned
            without attributes. If None uses the global default.

        Examples
        --------
        >>> da = xr.DataArray([1, 1, 2, 2, 2], dims="x")
        >>> da.rolling_exp(x=2, window_type="span").mean()
        <xarray.DataArray (x: 5)> Size: 40B
        array([1.        , 1.        , 1.69230769, 1.9       , 1.96694215])
        Dimensions without coordinates: x
        """
        pass

    def sum(self, keep_attrs: bool | None=None) -> T_DataWithCoords:
        """
        Exponentially weighted moving sum.

        Parameters
        ----------
        keep_attrs : bool, default: None
            If True, the attributes (``attrs``) will be copied from the original
            object to the new one. If False, the new object will be returned
            without attributes. If None uses the global default.

        Examples
        --------
        >>> da = xr.DataArray([1, 1, 2, 2, 2], dims="x")
        >>> da.rolling_exp(x=2, window_type="span").sum()
        <xarray.DataArray (x: 5)> Size: 40B
        array([1.        , 1.33333333, 2.44444444, 2.81481481, 2.9382716 ])
        Dimensions without coordinates: x
        """
        pass

    def std(self) -> T_DataWithCoords:
        """
        Exponentially weighted moving standard deviation.

        `keep_attrs` is always True for this method. Drop attrs separately to remove attrs.

        Examples
        --------
        >>> da = xr.DataArray([1, 1, 2, 2, 2], dims="x")
        >>> da.rolling_exp(x=2, window_type="span").std()
        <xarray.DataArray (x: 5)> Size: 40B
        array([       nan, 0.        , 0.67936622, 0.42966892, 0.25389527])
        Dimensions without coordinates: x
        """
        pass

    def var(self) -> T_DataWithCoords:
        """
        Exponentially weighted moving variance.

        `keep_attrs` is always True for this method. Drop attrs separately to remove attrs.

        Examples
        --------
        >>> da = xr.DataArray([1, 1, 2, 2, 2], dims="x")
        >>> da.rolling_exp(x=2, window_type="span").var()
        <xarray.DataArray (x: 5)> Size: 40B
        array([       nan, 0.        , 0.46153846, 0.18461538, 0.06446281])
        Dimensions without coordinates: x
        """
        pass

    def cov(self, other: T_DataWithCoords) -> T_DataWithCoords:
        """
        Exponentially weighted moving covariance.

        `keep_attrs` is always True for this method. Drop attrs separately to remove attrs.

        Examples
        --------
        >>> da = xr.DataArray([1, 1, 2, 2, 2], dims="x")
        >>> da.rolling_exp(x=2, window_type="span").cov(da**2)
        <xarray.DataArray (x: 5)> Size: 40B
        array([       nan, 0.        , 1.38461538, 0.55384615, 0.19338843])
        Dimensions without coordinates: x
        """
        pass

    def corr(self, other: T_DataWithCoords) -> T_DataWithCoords:
        """
        Exponentially weighted moving correlation.

        `keep_attrs` is always True for this method. Drop attrs separately to remove attrs.

        Examples
        --------
        >>> da = xr.DataArray([1, 1, 2, 2, 2], dims="x")
        >>> da.rolling_exp(x=2, window_type="span").corr(da.shift(x=1))
        <xarray.DataArray (x: 5)> Size: 40B
        array([       nan,        nan,        nan, 0.4330127 , 0.48038446])
        Dimensions without coordinates: x
        """
        pass