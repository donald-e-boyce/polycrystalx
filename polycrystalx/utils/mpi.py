"""MPI Utilities"""
from mpi4py import MPI


myrank = MPI.COMM_WORLD.rank
commsize = MPI.COMM_WORLD.size


def mpi_sync():
    """Sync all MPI processes using a collective barrier"""
    MPI.COMM_WORLD.Barrier()
