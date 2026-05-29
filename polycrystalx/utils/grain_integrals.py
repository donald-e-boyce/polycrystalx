"""Class for evaluation of grain integrals"""

import numpy as np
from mpi4py import MPI
from ufl import Measure, TestFunction
from dolfinx import fem


class GrainIntegrals:
    """Class for evaluating integrals over the grains of a microstructure

    Assembles per-cell integrals in a single vectorized pass using a DG(0)
    test function, then sums by grain.

    Parameters
    ----------
    msh: Mesh
        the mesh being used
    grain_cells: dict
        gives array of cell ids on this process for each grain
    """

    def __init__(self, msh,  grain_cells, comm=MPI.COMM_WORLD):
        self.grain_cells = grain_cells
        self.comm = comm

        Vdg = fem.functionspace(msh, ("DG", 0))
        self._n_local = Vdg.dofmap.index_map.size_local

        self.integrand = fem.Function(V)
        dx = Measure("dx", domain=msh)
        v = TestFunction(Vdg)
        self._cell_form = fem.form(v * self.integrand * dx)

    def grain_integrals(self, f):
        """Compute grain integrals of scalar function.

        Parameters
        ----------
        f : dolfinx Function
            Scalar function to integrate over the grains.

        Returns
        -------
        array
            Array of grain integrals, one entry per grain.
        """
        self.integrand.x.array[:] = f.x.array
        self.integrand.x.scatter_forward()

        b = fem.assemble_vector(self._cell_form)
        cell_integrals = b.array[:self._n_local]

        ng = len(self.grain_cells)
        integrals = np.zeros(ng)
        for g in range(ng):
            gcells = self.grain_cells[g]
            local_sum = float(cell_integrals[gcells].sum()) if len(gcells) > 0 else 0.0
            integrals[g] = self.comm.allreduce(local_sum, op=MPI.SUM)

        return integrals
