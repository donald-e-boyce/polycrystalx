"""This is the module for defining and executing model processes"""
import os
import pathlib

from dolfinx import log

from .linear_elasticity import LinearElasticity
from .heat_transfer import HeatTransfer

from ..utils.mpi import mpi_sync, myrank


processes = (LinearElasticity, HeatTransfer)
process_dict = {}
for p in processes:
    process_dict[p.name] = p


def run(job):
    """Run a job

    PARAMETERS
    ----------
    job: inputs.job.Job
       the job to run
    """
    outdir = setup_output(job.output_directory)
    process = process_dict[job.process](job)
    process.run(outdir)


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
