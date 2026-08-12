"""Mesh Input Module"""

import numpy as np

from polycrystalx import inputs


def get_mesh_input(key):
    """Return a named mesh input"""
    return MeshInput(key).mesh_input


class MeshInput:
    """Builds mesh input for meshx

    Parameters:
    ----------
    key: tuple
        mesh subdivisions in each direction
    """

    extents = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]

    def __init__(self, key):
        self.divs = key

    @property
    def name(self):
        return "x".join([str(d) for d in self.divs])

    @property
    def mesh_input(self):
        return inputs.mesh.Mesh(
            name=self.name,
            source="box",
            extents=self.extents,
            divisions=self.divs,
            celltype="tetrahedron",
            boundary_sections=[]
        )
