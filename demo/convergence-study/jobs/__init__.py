"""Inputs Module"""
from . import batch

# Material is orthotropic.
matl_key = "ort-235"

# Microstructure is the fifth 10-grain random Voronoi microstructure.
poly_key = ("voronoi", 10, 5)

# Mesh subdivisions are 50 in each direction.
mesh_key = 3 * (50,)

# Deformation has applied temperature on the top and bottom and a simple heat source.
defm_key = "temp-z-hs"

jobkey = (matl_key, poly_key, mesh_key, defm_key)
job = batch.get_job(jobkey)
