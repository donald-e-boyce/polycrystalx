"""Utilities for handling input"""
import os
import pathlib

import numpy as np

from dolfinx import log
from dolfinx.fem import assemble_scalar

from .xdmffile_ext import XDMFFile_Ext
from .mpi import MPI, mpi_sync, myrank

from .grain_integrals import GrainIntegrals


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
