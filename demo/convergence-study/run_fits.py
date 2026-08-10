"""Convergence postprocessing script.

Reads a YAML config, runs grain-average and continuous-grain convergence
analysis on each named suite, and saves all results to a single HDF5 file.
For Voronoi microstructures the Voronoi seed points are also used as query
points for a point-wise error analysis, and grain-interface (ridge) points
derived from the Voronoi geometry are evaluated as well.

Usage
-----
    mpirun -n <N> python run_convergence.py <config.yaml>

YAML format
-----------
    name: <str>          # used as the output filename: <name>.h5
    suites:              # list of attribute names in jobs.batch
      - voronoi_10g_1
      - voronoi_10g_2
      - ...

HDF5 layout
-----------
    /<suite_name>/
        h                                    # (njobs,) mesh-size parameters
        volume                               # (njobs, ngrains) grain volumes
        grain_avg/
            temperature/
                errors                       # (njobs-1, ngrains)
                rates                        # (ngrains,)
                intercepts                   # (ngrains,)
            flux/
                errors                       # (njobs-1, ngrains)
                rates                        # (ngrains,)
                intercepts                   # (ngrains,)
        continuous_grain/
            temperature/
                errors                       # (njobs-1, ngrains)
                rates                        # (ngrains,)
                intercepts                   # (ngrains,)
            flux/
                errors                       # (njobs-1, ngrains)
                rates                        # (ngrains,)
                intercepts                   # (ngrains,)
        point_errors/                        # Voronoi suites only
            seeds                            # (ngrains, 3) — Voronoi seed points
            temperature/
                errors                       # (njobs-1, ngrains)
                rates                        # (ngrains,)
                intercepts                   # (ngrains,)
            flux/
                errors                       # (njobs-1, ngrains)
                rates                        # (ngrains,)
                intercepts                   # (ngrains,)
        ridge_errors/                        # Voronoi suites only
            ridge_pairs                      # (n_ridges, 2) — neighboring seed index pairs
            ridge_points                     # (2*n_ridges, 3) — 45%/55% points per pair
            temperature/
                errors                       # (njobs-1, 2*n_ridges)
                rates                        # (2*n_ridges,)
                intercepts                   # (2*n_ridges,)
            flux/
                errors                       # (njobs-1, 2*n_ridges)
                rates                        # (2*n_ridges,)
                intercepts                   # (2*n_ridges,)
"""
import argparse
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import Voronoi as ScipyVoronoi
import yaml
from mpi4py import MPI

# Make sure the convergence-study directory is on sys.path so that
# `jobs` and `postprocess` are importable regardless of where the
# script is invoked from.
sys.path.insert(0, str(Path(__file__).parent))

import postprocess
from jobs import batch

logger = logging.getLogger(__name__)

TEMP = "temperature"
FLUX = "flux"


def fit_per_grain(errors, h):
    """Fit log-log convergence for each grain column in *errors*.

    Parameters
    ----------
    errors : ndarray, shape (njobs, ngrains)
    h : ndarray, shape (njobs,)

    Returns
    -------
    rates : ndarray, shape (ngrains,)
    intercepts : ndarray, shape (ngrains,)
    """
    ngrains = errors.shape[1]
    rates = np.zeros(ngrains)
    intercepts = np.zeros(ngrains)
    for g in range(ngrains):
        rates[g], intercepts[g] = postprocess.Suite.logfit(errors[:, g], h)
    return rates, intercepts


def get_ridge_points(seeds):
    """Compute grain-interface evaluation points from Voronoi ridge pairs.

    Uses scipy.spatial.Voronoi to identify neighboring seed pairs, then
    returns two points per pair: one at 45% and one at 55% of the way from
    seeds[i] to seeds[j], straddling the grain interface at the midpoint.

    Parameters
    ----------
    seeds : ndarray, shape (n, 3)
        Voronoi seed points.

    Returns
    -------
    ridge_pairs : ndarray, shape (n_ridges, 2)
        Seed index pairs for each Voronoi ridge.
    ridge_pts : ndarray, shape (2*n_ridges, 3)
        Evaluation points interleaved per ridge: row 2k is the 45% point
        and row 2k+1 is the 55% point for ridge k.
    """
    vor = ScipyVoronoi(seeds)
    pairs = vor.ridge_points                 # (n_ridges, 2)

    si = seeds[pairs[:, 0]]
    sj = seeds[pairs[:, 1]]
    d = sj - si

    pts_45 = si + 0.45 * d
    pts_55 = si + 0.55 * d

    n_ridges = len(pairs)
    ridge_pts = np.empty((2 * n_ridges, 3))
    ridge_pts[0::2] = pts_45
    ridge_pts[1::2] = pts_55

    return pairs, ridge_pts


def process_suite(suite_name, h5root):
    """Run full convergence analysis for one suite and write into *h5root*.

    Parameters
    ----------
    suite_name : str
        Attribute name in ``jobs.batch`` (e.g. ``"voronoi_10g_1"``).
    h5root : h5py.File or h5py.Group
        Open HDF5 handle; results are written under a group named
        *suite_name*.
    """
    logger.info(f"=== suite: {suite_name} ===")

    # Retrieve the job-key iterable and force it to a list immediately so
    # that itertools.product objects (single-use) are fully materialised.
    job_keys = list(getattr(batch, suite_name))
    logger.info(f"  {len(job_keys)} jobs")

    suite = postprocess.Suite(job_keys)
    avg_data = suite.average_data          # triggers load of all .npz files
    h = avg_data.h                         # (njobs,)
    h_coarse = h[:-1]                      # used for fitting (exclude finest)

    grp = h5root.require_group(suite_name)
    grp.create_dataset("h", data=h)
    grp.create_dataset("volume", data=avg_data.volume)   # (njobs, ngrains)

    # ------------------------------------------------------------------ #
    # Grain-average convergence                                            #
    # ------------------------------------------------------------------ #
    ga_grp = grp.require_group("grain_avg")

    for field in (TEMP, FLUX):
        rates, intercepts = suite.fit_average_errors(field)

        fld_data = getattr(avg_data, field)
        errors = suite.average_errors(fld_data)          # (njobs-1, ngrains)

        fg = ga_grp.require_group(field)
        fg.create_dataset("errors", data=errors)
        fg.create_dataset("rates", data=rates)
        fg.create_dataset("intercepts", data=intercepts)

        logger.info(f"  grain_avg/{field}: mean rate = {rates.mean():.3f}")

    # ------------------------------------------------------------------ #
    # Continuous grain errors                                              #
    # ------------------------------------------------------------------ #
    logger.info("  computing continuous grain errors …")
    errs_temp, errs_flux = suite.get_continuous_grain_errors()

    cg_grp = grp.require_group("continuous_grain")

    for field, errors in ((TEMP, errs_temp), (FLUX, errs_flux)):
        rates, intercepts = fit_per_grain(errors, h_coarse)

        fg = cg_grp.require_group(field)
        fg.create_dataset("errors", data=errors)
        fg.create_dataset("rates", data=rates)
        fg.create_dataset("intercepts", data=intercepts)

        logger.info(f"  continuous_grain/{field}: mean rate = {rates.mean():.3f}")

    # ------------------------------------------------------------------ #
    # Point-wise errors at Voronoi seed points                            #
    # ------------------------------------------------------------------ #
    ms = suite.jobs[0].polycrystal_input.polycrystal
    seeds = getattr(ms, "seeds", None)

    if seeds is not None:
        logger.info("  computing point-wise errors at Voronoi seeds …")
        pt_errors = suite.get_point_errors(seeds)

        pe_grp = grp.require_group("point_errors")
        pe_grp.create_dataset("seeds", data=seeds)

        for field, errors in ((TEMP, pt_errors.temperature),
                              (FLUX, pt_errors.flux)):
            rates, intercepts = fit_per_grain(errors, h_coarse)

            fg = pe_grp.require_group(field)
            fg.create_dataset("errors",     data=errors)
            fg.create_dataset("rates",      data=rates)
            fg.create_dataset("intercepts", data=intercepts)

            logger.info(f"  point_errors/{field}: mean rate = {rates.mean():.3f}")

    # ------------------------------------------------------------------ #
    # Point-wise errors at grain interface (ridge) points                  #
    # ------------------------------------------------------------------ #
    if seeds is not None:
        logger.info("  computing point-wise errors at grain interface points …")
        ridge_pairs, ridge_pts = get_ridge_points(seeds)
        logger.info(f"  {len(ridge_pairs)} Voronoi ridges → {len(ridge_pts)} interface points")
        pt_errors_ridge = suite.get_point_errors(ridge_pts)

        rp_grp = grp.require_group("ridge_errors")
        rp_grp.create_dataset("ridge_pairs",  data=ridge_pairs)
        rp_grp.create_dataset("ridge_points", data=ridge_pts)

        for field, errors in ((TEMP, pt_errors_ridge.temperature),
                              (FLUX, pt_errors_ridge.flux)):
            rates, intercepts = fit_per_grain(errors, h_coarse)

            fg = rp_grp.require_group(field)
            fg.create_dataset("errors",     data=errors)
            fg.create_dataset("rates",      data=rates)
            fg.create_dataset("intercepts", data=intercepts)

            logger.info(
                f"  ridge_errors/{field}: mean rate = {rates.mean():.3f}"
            )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Run convergence postprocessing and save results to HDF5."
    )
    parser.add_argument("config", help="path to YAML configuration file")
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    name = cfg["name"]
    suite_names = cfg["suites"]
    output_path = Path(name + ".h5")

    rank = MPI.COMM_WORLD.rank

    if rank == 0:
        logger.info(f"config:  {config_path}")
        logger.info(f"name:    {name}")
        logger.info(f"suites:  {suite_names}")
        logger.info(f"output:  {output_path}")

    # Open the HDF5 file on rank 0 only; the heavy FEniCSx work uses MPI
    # internally, but the numpy results are the same on every rank.
    if rank == 0:
        hf = h5py.File(output_path, "w")
    else:
        hf = None

    try:
        for suite_name in suite_names:
            process_suite(suite_name, hf if rank == 0 else _NullGroup())
    finally:
        if rank == 0:
            hf.close()
            logger.info(f"saved → {output_path}")


class _NullGroup:
    """Drop-in for h5py.File/Group on non-root MPI ranks.

    All write calls are silently ignored so that non-root ranks can
    execute the same code path without touching the file.
    """
    def require_group(self, name):
        return self

    def create_dataset(self, name, **kwargs):
        pass


if __name__ == "__main__":
    main()
