from __future__ import annotations
import functools
import sys
from itertools import repeat
from typing import TYPE_CHECKING, Callable
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset
from xarray.core.formatting import diff_treestructure
from xarray.core.treenode import NodePath, TreeNode
if TYPE_CHECKING:
    from xarray.core.datatree import DataTree

class TreeIsomorphismError(ValueError):
    """Error raised if two tree objects do not share the same node structure."""
    pass

def check_isomorphic(a: DataTree, b: DataTree, require_names_equal: bool=False, check_from_root: bool=True):
    """
    Check that two trees have the same structure, raising an error if not.

    Does not compare the actual data in the nodes.

    By default this function only checks that subtrees are isomorphic, not the entire tree above (if it exists).
    Can instead optionally check the entire trees starting from the root, which will ensure all

    Can optionally check if corresponding nodes should have the same name.

    Parameters
    ----------
    a : DataTree
    b : DataTree
    require_names_equal : Bool
        Whether or not to also check that each node has the same name as its counterpart.
    check_from_root : Bool
        Whether or not to first traverse to the root of the trees before checking for isomorphism.
        If a & b have no parents then this has no effect.

    Raises
    ------
    TypeError
        If either a or b are not tree objects.
    TreeIsomorphismError
        If a and b are tree objects, but are not isomorphic to one another.
        Also optionally raised if their structure is isomorphic, but the names of any two
        respective nodes are not equal.
    """
    pass

def map_over_subtree(func: Callable) -> Callable:
    """
    Decorator which turns a function which acts on (and returns) Datasets into one which acts on and returns DataTrees.

    Applies a function to every dataset in one or more subtrees, returning new trees which store the results.

    The function will be applied to any data-containing dataset stored in any of the nodes in the trees. The returned
    trees will have the same structure as the supplied trees.

    `func` needs to return one Datasets, DataArrays, or None in order to be able to rebuild the subtrees after
    mapping, as each result will be assigned to its respective node of a new tree via `DataTree.__setitem__`. Any
    returned value that is one of these types will be stacked into a separate tree before returning all of them.

    The trees passed to the resulting function must all be isomorphic to one another. Their nodes need not be named
    similarly, but all the output trees will have nodes named in the same way as the first tree passed.

    Parameters
    ----------
    func : callable
        Function to apply to datasets with signature:

        `func(*args, **kwargs) -> Union[DataTree, Iterable[DataTree]]`.

        (i.e. func must accept at least one Dataset and return at least one Dataset.)
        Function will not be applied to any nodes without datasets.
    *args : tuple, optional
        Positional arguments passed on to `func`. If DataTrees any data-containing nodes will be converted to Datasets
        via `.ds`.
    **kwargs : Any
        Keyword arguments passed on to `func`. If DataTrees any data-containing nodes will be converted to Datasets
        via `.ds`.

    Returns
    -------
    mapped : callable
        Wrapped function which returns one or more tree(s) created from results of applying ``func`` to the dataset at
        each node.

    See also
    --------
    DataTree.map_over_subtree
    DataTree.map_over_subtree_inplace
    DataTree.subtree
    """
    pass

def _handle_errors_with_path_context(path: str):
    """Wraps given function so that if it fails it also raises path to node on which it failed."""
    pass

def _check_single_set_return_values(path_to_node: str, obj: Dataset | DataArray | tuple[Dataset | DataArray]):
    """Check types returned from single evaluation of func, and return number of return values received from func."""
    pass

def _check_all_return_values(returned_objects):
    """Walk through all values returned by mapping func over subtrees, raising on any invalid or inconsistent types."""
    pass