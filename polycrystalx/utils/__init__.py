"""Utilities

These are simple utilities that are not necessarily specific to any particular top
level package, but are generally useful or needed for pre- or post-processing.


Routine Listings
----------------

grain_integrals(f, grain_cells, comm=MPI.COMM_WORLD, symmetric=True)
    compute grain integrals

interpolate_to(f, V_hat, padding = 1e-10):
    interpolate a function onto another mesh

EvalPoints
    class for pointwise evaluation of functions
"""
import os
import pathlib

import numpy as np

from dolfinx import fem, log, geometry
from ufl import Measure, TestFunction

from .xdmffile_ext import XDMFFile_Ext
from .mpi import MPI, mpi_sync, myrank


SYMMETRIC_INDICES = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
SYMMETRIC_ISUBS = [0, 4, 8, 5, 2, 1]


def grain_integrals(f, grain_cells, comm=MPI.COMM_WORLD, symmetric=True):
    """Evaluate the grain integrals of a function over a  microstructure

    Parameters
    ----------
    f: Function
        the function to integrate; it can have scalar, vector or tensor values
    grain_cells: dict
        gives array of cell ids on this process for each grain
    comm: MPI communicator
        the MPI communicator
    symmetric: bool
        if True, and shape is (3, 3), then only integrate the six subfunctions with
        indices (0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (1, 2)
    """
    ng = len(grain_cells)
    shp = f.ufl_shape
    # If scalar, the just return the integral. Otherwise, integrate each index.
    if len(shp) == 0:
        return _scalar_grain_integrals(f, grain_cells)

    if shp == (3, 3) and symmetric:
        indices = SYMMETRIC_INDICES
        isubs = SYMMETRIC_ISUBS
        integrals = np.zeros((ng, 6))
    else:
        indices = list(np.ndindex(shp))
        isubs = np.arange(len(indices))
        integrals = np.zeros((ng,) + shp)

    for i, (index, isub) in enumerate(zip(indices, isubs)):

        slc = (slice(None), i) if symmetric else  (slice(None),) + index
        integrals[slc] = (
            _scalar_grain_integrals(f.sub(isub), grain_cells)
        )

    return integrals


def _scalar_grain_integrals(f, grain_cells, comm=MPI.COMM_WORLD):
    """Evaluate the grain integrals of a function over a  microstructure

    Assembles per-cell integrals in a single vectorized pass using a DG(0)
    test function, then sums by grain.

    Parameters
    ----------
    f: Function
        a scalar-valued function to integrate
    grain_cells: dict
        gives array of cell ids on this process for each grain
    """

    msh = f.function_space.mesh
    Vdg = fem.functionspace(msh, ("DG", 0))
    _n_local = Vdg.dofmap.index_map.size_local

    dx = Measure("dx", domain=msh)
    v = TestFunction(Vdg)
    _cell_form = fem.form(v * f * dx)

    b = fem.assemble_vector(_cell_form)
    cell_integrals = b.array[:_n_local]

    ng = len(grain_cells)
    integrals = np.zeros(ng)
    for g in range(ng):
        gcells = grain_cells[g]
        local_sum = float(cell_integrals[gcells].sum()) if len(gcells) > 0 else 0.0
        integrals[g] = comm.allreduce(local_sum, op=MPI.SUM)

    return integrals


def interpolate_to(f, V_hat, padding = 1e-10):
    """Interpolate a function onto another mesh

    Parameters
    ----------
    f: Function
       a dolfinx Function on function space V
    V_hat: FunctionSpace
       the function space to which to interpolate

    Returns
    -------
    f_hat: Function
       the function `f` interpolated onto V_hat
    """
    V = f.function_space
    f_hat = fem.Function(V_hat)

    # Get cells on mesh to interpolate.
    msh = V_hat.mesh
    vhat_cell_map = msh.topology.index_map(msh.topology.dim)
    num_cells_on_proc = vhat_cell_map.size_local + vhat_cell_map.num_ghosts
    cells = np.arange(num_cells_on_proc, dtype=np.int32)

    interpolation_data = fem.create_interpolation_data(
        V_hat, V, cells, padding=padding
    )
    f_hat.interpolate_nonmatching(f, cells, interpolation_data=interpolation_data)

    return f_hat


class EvalPoints:
    """Evaluate dolfinx Functions at a fixed set of points

    The bounding-box tree and cell-collision search are performed once in
    ``__init__``; subsequent calls to :meth:`eval` reuse those results so
    that many fields on the same mesh can be sampled without repeating the
    geometry lookup.

    Parameters
    ----------
    points : np.ndarray, shape (n, 3)
        Query points; must be identical on every MPI rank.
    msh : dolfinx.mesh.Mesh
        Mesh on which functions will be evaluated.
    """

    def __init__(self, points, msh):
        self._points = np.asarray(points, dtype=np.float64)
        bbt = geometry.bb_tree(msh, msh.topology.dim)
        cell_candidates = geometry.compute_collisions_points(bbt, self._points)
        colliding_cells = geometry.compute_colliding_cells(
            msh, cell_candidates, self._points
        )

        indices = []
        cells = []
        for i in range(len(self._points)):
            links = colliding_cells.links(i)
            if len(links) > 0:
                indices.append(i)
                cells.append(links[0])

        self._indices = np.array(indices, dtype=np.intp)
        self._cells = cells

    def eval(self, f, return_status=False, tol=1e-8):
        """Evaluate Function *f* at the query points on all MPI ranks.

        Shares per-rank ``(indices, values)`` pairs via ``allgather`` and
        assembles a complete result array in a single O(n) pass.  If a point
        is found on more than one rank its value is the mean of the per-rank
        values (provided they agree within *tol*); otherwise ``NaN``.

        Parameters
        ----------
        f : dolfinx.fem.Function
            Function to evaluate; must live on the mesh passed to ``__init__``.
        return_status : bool, optional
            When True a status array is returned alongside the values.
            Default False.
        tol : float, optional
            Maximum absolute deviation from the mean allowed when merging
            values found on multiple ranks.  Default 1e-8.

        Returns
        -------
        result : np.ndarray
            Shape ``(n,)`` for scalar functions, ``(n, bs)`` for functions
            with block size *bs*.  Entries are ``NaN`` for points not found
            in the domain.
        status : np.ndarray of int, shape (n,)  [only when return_status=True]
            ``0``  — point not found on any rank (result entry is ``NaN``).
            ``k``  — point found on *k* ranks; result is the mean value if
                     all per-rank values agree within *tol*, otherwise ``NaN``.
        """
        from collections import defaultdict

        comm = MPI.COMM_WORLD
        n = len(self._points)

        # Per-rank evaluation using cached indices and cells.
        if len(self._indices) > 0:
            vals_local = f.eval(self._points[self._indices], self._cells)
            # dolfinx may squeeze a single-point vector eval to 1-D, e.g.
            # shape (3,) instead of (1, 3).  Normalise to (n_found, value_size)
            # so that allgather always produces uniform 2-D arrays.
            if vals_local.ndim == 1:
                vals_local = vals_local.reshape(len(self._indices), -1)
        else:
            bs = f.function_space.dofmap.bs
            vals_local = np.empty((0, bs), dtype=np.float64)

        # Share all per-rank (index, value) pairs with every rank.
        all_idx_list  = comm.allgather(self._indices)
        all_vals_list = comm.allgather(vals_local)

        # Block size determines scalar vs. vector layout.
        bs = f.function_space.dofmap.bs
        is_scalar = (bs == 1)

        if n == 0:
            result = np.empty(0) if is_scalar else np.empty((0, bs))
            status = np.zeros(0, dtype=int)
            return (result, status) if return_status else result

        result = np.full(n, np.nan) if is_scalar else np.full((n, bs), np.nan)
        status = np.zeros(n, dtype=int)

        # Collect per-rank values keyed by point index — O(total found points).
        found = defaultdict(list)
        for r_idx, r_vals in zip(all_idx_list, all_vals_list):
            for k, i in enumerate(r_idx):
                if is_scalar:
                    found[i].append(float(np.ravel(r_vals[k])[0]))
                else:
                    found[i].append(np.ravel(r_vals[k]).astype(float))

        # Assemble result — O(n).
        for i, vals_list in found.items():
            count = len(vals_list)
            status[i] = count
            if count == 1:
                result[i] = vals_list[0]
            else:
                arr = np.array(vals_list)           # (count,) or (count, bs)
                mean_val = arr.mean(axis=0)
                if np.all(np.abs(arr - mean_val) <= tol):
                    result[i] = float(mean_val) if is_scalar else mean_val
                # else NaN stays (values disagree beyond tol)

        return (result, status) if return_status else result
