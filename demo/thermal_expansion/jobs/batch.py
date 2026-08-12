"""Batch jobs"""

from polycrystalx import inputs

from .job_inputs import (
    get_material_input,
    get_microstructure_input,
    get_mesh_input,
    get_deformation_input,
)

suite = "thermal_expansion"
process = "linear-elasticity"

# The `get_job` function is required for running batch jobs.

def get_job(key):
    matl, mesh, poly, defm = key
    matl_input = get_material_input(matl)
    mesh_input = get_mesh_input(mesh)
    poly_input = get_microstructure_input(poly)
    defm_input = get_deformation_input(defm, matl_input)

    return inputs.job.Job(
        suite = suite,
        process = process,
        mesh_input = mesh_input,
        material_input = matl_input,
        polycrystal_input = poly_input,
        deformation_input = defm_input
    )


# Next, we set up the test suite.

ident = "identity-iso"
ti64 = "ti-64-bar-RT"
poly_key = None
mesh_key = (20, 20, 20)
defm_keys = ["zero", "match", "linear"]

job_keys = [
    (ident, mesh_key, poly_key, "zero"),
    (ident, mesh_key, poly_key, "match"),
    (ti64, mesh_key, poly_key, "linear"),
]
