"""Batch jobs"""
import itertools

from .job_inputs import get_job


# Define suites of jobs. Each job needs four inputs: material, microstructure, mesh
# and deformation.  Each input has a corresponding `key` that is used to generate that
# particular input.


# Mesh sequences.

mesh_keys_small = [3 * (10 * i,) for i in range(2, 6)] + [3 * (100,)] # 5 meshes
mesh_keys_full = [3 * (10 * i,) for i in range(2, 11)] + [3 * (200,)] # 10 meshes


def voronoi_jobs(i, mesh_keys):
    """Return job-keys list for convergence suite `i`

    Parameters
    ----------
    i: int
      i'th random Voronoi microstructure
    mesh_keys: list
      list of mesh keys for meshes used in study

    Returns
    -------
    list:
       list of job-key tuples for running a particular Voronoi microstructure with
       a sequence of meshes
    """
    return itertools.product(
        [ "ort-235"],
        [("voronoi", 10, i)],
        mesh_keys,
        ["temp-z-hs"]
    )


# Full mesh study.

voronoi_10g_1 = voronoi_jobs(1, mesh_keys_full)
voronoi_10g_2 = voronoi_jobs(2, mesh_keys_full)
voronoi_10g_3 = voronoi_jobs(3, mesh_keys_full)
voronoi_10g_4 = voronoi_jobs(4, mesh_keys_full)
voronoi_10g_5 = voronoi_jobs(5, mesh_keys_full)

voronoi_10g = itertools.chain(
    voronoi_10g_1,
    voronoi_10g_2,
    voronoi_10g_3,
    voronoi_10g_4,
    voronoi_10g_5,
)


# Small mesh study for testing.

voronoi_10g_1_small = voronoi_jobs(1, mesh_keys_small)
voronoi_10g_2_small = voronoi_jobs(2, mesh_keys_small)
voronoi_10g_3_small = voronoi_jobs(3, mesh_keys_small)
voronoi_10g_4_small = voronoi_jobs(4, mesh_keys_small)
voronoi_10g_5_small = voronoi_jobs(5, mesh_keys_small)

voronoi_10g_small = itertools.chain(
    voronoi_10g_1_small,
    voronoi_10g_2_small,
    voronoi_10g_3_small,
    voronoi_10g_4_small,
    voronoi_10g_5_small,
)
