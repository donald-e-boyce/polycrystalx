"""Fenicsx utilities

These are a couple of useful functions that will eventually get moved to the
main polycrystalx repository.
"""

import numpy as np
from mpi4py import MPI

from ufl import Measure, TestFunction, grad, inner
from dolfinx import fem, geometry, mesh as dmesh
from dolfinx.fem import assemble_scalar


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
