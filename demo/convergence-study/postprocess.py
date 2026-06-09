"""Postprocessing for convergence study"""

from collections import namedtuple
import logging

import numpy as np
from mpi4py import MPI
import dolfinx
from ufl import Measure, grad
from dolfinx import fem

from polycrystalx.loaders.polycrystal import Polycrystal as PxLoader
from polycrystalx.utils import XDMFFile_Ext, grain_integrals
from fenicsx_utils import interpolate_to, EvalPoints


from jobs import batch


logger = logging.getLogger(__name__)

TEMP, FLUX, VOL, H = "temperature", "flux", "volume", "h"
flds = [TEMP, FLUX, VOL, H]
AverageData = namedtuple("AverageData", flds)
AverageData.__doc__ = """Class for holding grain average data for all jobs

Parameters
----------
temperature: array(njobs, ngrains)
  array of grain-averaged temperatures over all jobs
flux: array(njobs, ngrains, 3)
  array of grain-averaged fluxs of grain over all jobs
volume: array(njobs, ngrains)
  array of grain volumes over all jobs
h: array(njobs, ngrains)
  array of mesh size parameters over all jobs
"""


flds = ["mesh", "mesh_tags", "temperature", "flux"]
JobSolution = namedtuple("JobSolution", flds)
JobSolution.__doc__ = """Class for holidng FE solution data
"""

PointData = namedtuple("PointData", ["points", "temperature", "flux"])
PointData.__doc__ = """Point-evaluation data for all jobs in a suite.

Parameters
----------
points      : ndarray (n, 3)
temperature : ndarray (num_jobs, n)        — temperature at each point/job
flux        : ndarray (num_jobs, n, 3)     — flux vector at each point/job
"""

PointErrors = namedtuple("PointErrors", ["temperature", "flux"])
PointErrors.__doc__ = """Point-wise errors vs the finest-mesh reference solution.

Parameters
----------
temperature : ndarray (num_jobs-1, n)      — |T_job − T_ref| at each point
flux        : ndarray (num_jobs-1, n)      — ‖F_job − F_ref‖ at each point
"""

del flds

np.set_printoptions(precision=3)




class Suite:
    """Postprocessing for a job suite in a convergence study.

    Loads grain-averaged outputs for each job in the suite and computes
    errors relative to the finest mesh solution.

    Parameters
    ----------
    jkeys: iterator
        Iterator over job keys identifying each job in the suite.
    """

    def __init__(self, jkeys):
        self.job_keys = list(jkeys)
        self.num_jobs = len(self.job_keys)
        self.jobs = [batch.get_job(jk) for jk in self.job_keys]

    @property
    def average_data(self):
        """Grain-averaged data for all jobs, computed once and cached.

        Returns
        -------
        AverageData
            Named tuple with arrays of shape ``(num_jobs, num_grains)`` for
            temperature and flux, a volume array, and a 1-D array of mesh
            size parameters ``h``.
        """
        if not hasattr(self, "_average_data"):
            self._average_data = self._get_average_data()
        return self._average_data

    def _get_average_data(self):
        """Load grain-averaged outputs for all jobs and assemble into arrays.

        Returns
        -------
        AverageData
            Named tuple with arrays stacked over jobs.
        """
        gdata = [self.load_averages(j) for j in self.jobs]

        temp = np.stack([gz[TEMP] for gz in gdata])
        flux = np.stack([gz[FLUX] for gz in gdata])
        volume = np.stack([gz[VOL] for gz in gdata])
        self.h = h = np.array([self.get_h(j) for j in self.jobs])

        return AverageData(temp, flux, volume, h)

    @property
    def num_grains(self):
        return self.average_data.volume.shape[1]

    def job_solution(self, i):
        """Load the FEM solution for job ``i`` from its XDMF output file.

        Parameters
        ----------
        i: int
            Index into ``self.job_keys``; ``-1`` gives the finest-mesh job.

        Returns
        -------
        JobSolution
            Named tuple containing the mesh and temperature/flux functions.
        """
        jkeys = self.job_keys[i]
        job = batch.get_job(jkeys)

        fname = job.output_directory / "output.xdmf"
        logger.info(f"{fname=}")
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, fname, "r") as f:
            msh = f.read_mesh()
            mtags = f.read_meshtags(msh, "mesh_tags")
        with XDMFFile_Ext(msh.comm, fname, "r") as xfile:
            temp = xfile.read_function(msh, "f")
            flux = xfile.read_function(msh, "flux")
        return JobSolution(msh, mtags, temp, flux)

    @property
    def true_solution(self):
        """FEM solution from the finest mesh, loaded once and cached."""
        if not hasattr(self, "_true_solution"):
            self._true_solution = self.job_solution(-1)
        return self._true_solution

    def get_continuous_grain_errors(self):
        """Compute L2 errors for temperature/flux relative to the finest-mesh solution.

        Interpolates each job's fields onto the finest mesh.  Temperature error
        is the L2 norm of the scalar difference; flux error is the L2 norm of
        the Euclidean (vector) difference.

        Returns
        -------
        errs_temp : ndarray, shape (njobs, ngrains)
        errs_flux : ndarray, shape (njobs, ngrains)
        """
        ts = self.true_solution
        u = ts.temperature
        u_flux = ts.flux
        logger.info(f"max true_temp: {u.x.array.max()}")
        V_fine = u.function_space
        V_flux_fine = u_flux.function_space
        V_flux_fine_scalar = V_flux_fine.sub(0).collapse()[0]
        bs = V_flux_fine.dofmap.bs

        njobs = self.num_jobs - 1
        ngrns = self.num_grains
        errs_temp = np.zeros((njobs, ngrns))
        errs_flux = np.zeros((njobs, ngrns))
        logger.info(f"num-jobs, num-grains:  {njobs} {ngrns}")

        ldr = PxLoader(self.jobs[0].polycrystal_input)
        gcells = ldr.grain_cell_dict(ldr.grain_cell_tags(V_fine.mesh))

        umv_sq = fem.Function(V_fine)
        umv_sq_flux = fem.Function(V_flux_fine_scalar)

        for i in range(njobs):
            logger.info(f"job {i}:")
            si = self.job_solution(i)
            logger.info("... loaded")

            logger.info("... interpolating temperature to fine mesh")
            v = interpolate_to(si.temperature, V_fine)
            umv_sq.x.array[:] = (u.x.array - v.x.array) ** 2
            umv_sq.x.scatter_forward()
            errs_temp[i] = np.sqrt(grain_integrals(umv_sq, gcells))

            logger.info("... interpolating flux to fine mesh")
            w = interpolate_to(si.flux, V_flux_fine)
            d = (u_flux.x.array - w.x.array).reshape(-1, bs)
            umv_sq_flux.x.array[:] = (d ** 2).sum(axis=1)
            umv_sq_flux.x.scatter_forward()
            errs_flux[i] = np.sqrt(grain_integrals(umv_sq_flux, gcells))

            logger.info("... finished grain integrals")

        return errs_temp, errs_flux


    def eval_points(self, points):
        """Evaluate temperature and flux at *points* for every job solution.

        Parameters
        ----------
        points : ndarray, shape (n, 3)

        Returns
        -------
        PointData
            Named tuple whose fields are valid on every MPI rank:

            * ``points``      — the original query points, shape (n, 3)
            * ``temperature`` — shape (num_jobs, n)
            * ``flux``        — shape (num_jobs, n, 3)
        """
        n = len(points)
        temp_all = np.full((self.num_jobs, n), np.nan)
        flux_all = np.full((self.num_jobs, n, 3), np.nan)

        for i in range(self.num_jobs):
            logger.info(f"eval_points: job {i}")
            si = self.job_solution(i)
            ep = EvalPoints(points, si.temperature.function_space.mesh)
            temp_all[i] = ep.eval(si.temperature)
            flux_all[i] = ep.eval(si.flux)
            del si

        return PointData(points, temp_all, flux_all)

    def get_point_errors(self, points):
        """Compute temperature and flux errors at *points* vs the finest mesh.

        Parameters
        ----------
        points : ndarray, shape (n, 3)

        Returns
        -------
        PointErrors
            Named tuple whose fields are valid on every MPI rank:

            * ``temperature`` — shape (num_jobs-1, n),  absolute scalar error
            * ``flux``        — shape (num_jobs-1, n),  Euclidean vector error
        """
        pd = self.eval_points(points)

        temp_ref = pd.temperature[-1]        # (n,)
        flux_ref = pd.flux[-1]               # (n, 3)

        njobs = self.num_jobs - 1
        errs_temp = np.abs(pd.temperature[:njobs] - temp_ref)               # (njobs, n)
        errs_flux = np.linalg.norm(pd.flux[:njobs] - flux_ref, axis=-1)     # (njobs, n)

        return PointErrors(errs_temp, errs_flux)

    def load_averages(self, job):
        """Load the ``grain-averages.npz`` file for a single job.

        Parameters
        ----------
        job:
            Job object with an ``output_directory`` attribute.

        Returns
        -------
        NpzFile
            Loaded npz archive keyed by field name.
        """
        return np.load(job.output_directory / "grain-averages.npz")

    def get_h(self, job):
        """Return the mesh size parameter ``h`` for a job.

        Assumes uniform divisions in each direction; ``h = 1 / nx``.

        Parameters
        ----------
        job:
            Job object with a ``mesh_input.divisions`` attribute.
        """
        return 1.0 / job.mesh_input.divisions[0]

    def fit_average_errors(self, field):
        """Fit grain-averaged errors to a log-log model and print results.

        The model is ``err ~ c * h^n``, so a linear fit in log10 space gives
        the convergence rate ``n``.

        Parameters
        ----------
        field: {"temperature", "flux"}
            Which grain-averaged field to analyse.
        """
        data = self.average_data
        fld_data = getattr(data, field)
        errors = self.average_errors(fld_data)
        h = data.h[:-1]

        logger.info(field)
        logger.info(f"errors/h shape:  {errors.shape} {h.shape}")
        ma = np.zeros(self.num_grains)
        ba = np.zeros(self.num_grains)
        for g in range(self.num_grains):
            logger.info(f"grain: {g}")
            ma[g], ba[g] = m, b = self.logfit(errors[:, g], h)
            logger.info(f"log fit: m={m:.4f}, b={b:.4f}")

        return ma, ba

    @staticmethod
    def logfit(errors, h):
        """Fit errors vs mesh size to ``log10(err) = n*log10(h) + log10(c)``.

        Parameters
        ----------
        errors: array_like, shape (N,)
            Error values at each mesh resolution.
        h: array_like, shape (N,)
            Corresponding mesh size parameters.

        Returns
        -------
        m: float
            Convergence rate (slope in log-log space).
        b: float
            Log10 of the prefactor constant.
        """
        x = np.log10(h)
        y = np.log10(errors)
        m, b = np.polyfit(x, y, 1)
        return m, b

    def convergence(self):
        """Print grain-averaged fields and their errors vs the finest mesh."""
        data = self.average_data
        logger.info(f"flux\n{data.flux}")
        logger.info(f"  errors:\n{self.errors(data.flux)}")

        logger.info(f"temperature\n{data.temperature}")
        logger.info(f"  errors:\n{self.errors(data.temperature)}")

        logger.info("global temperature")

    @staticmethod
    def average_errors(fld, norm="l2"):
        """Absolute difference between each job's field and the finest-mesh value.

        Parameters
        ----------
        fld: ndarray, shape (num_jobs, num_grains)
            Stacked grain-averaged field values; last row is the reference.
        norm: {"l2", "max"}
            which norm to use for the error

        Returns
        -------
        ndarray, shape (num_jobs - 1, num_grains)
        """
        errs = np.abs(fld[:-1] - fld[-1])
        axes = tuple(range(2, errs.ndim))

        if norm == "max":
            nrm_errs = np.abs(errs).max(axis=axes)

        elif norm == "l2":
            if errs.ndim == 2:
                nrm_errs = np.abs(errs)
            else:
                nrm_errs = np.linalg.norm(errs, axis=-1)

        return nrm_errs
