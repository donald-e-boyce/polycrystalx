"""Utilities for handling input"""
import os
import pathlib

import numpy as np

from dolfinx import fem, log
from ufl import Measure, TestFunction

from .xdmffile_ext import XDMFFile_Ext
from .mpi import MPI, mpi_sync, myrank


SYMMETRIC_INDICES = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
SYMMETRIC_ISUBS = [0, 4, 8, 5, 2, 1]


def setup_output(outdir):
    """Make output directory if needed and return it as a Path.

    Parameters
    ----------
    outdir: str or Path
        name of output directory

    Returns
    -------
    pathlib.Path
        resolved output directory path
    """
    outdir = pathlib.Path(outdir)
    print("output directory: ", outdir)
    mpi_sync()
    if not os.path.exists(outdir):
        if myrank == 0:
            log.log(log.LogLevel.INFO, f"creating output directory: {outdir}")
            os.makedirs(outdir)
        mpi_sync()
    return outdir


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
