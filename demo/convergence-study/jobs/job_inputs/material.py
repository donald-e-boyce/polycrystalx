"""Material Input Module"""

import numpy as np

from polycrystalx import inputs

from polycrystal.heat_transfer.single_crystal import SingleCrystal


matl_dict = {
    "ort-235": SingleCrystal("orthotropic", (2.0, 3.0, 5.0)),
    "ort-111": SingleCrystal("orthotropic", (1.0, 1.0, 1.0)),
}


def get_material_input(key):
    """Return a named material input list"""
    return MaterialInput(key).material_input


class MaterialInput:
    """Builds material input for polycrystalx

    Parameters:
    ----------
    key: str (or list of strings)
       name of material
    """

    def __init__(self, key):
        self.key = key

    @property
    def name(self):
        return self.key

    @property
    def material_input(self):
        return inputs.material.LinearElasticity(
            name=self.name,
            materials=self.materials
        )

    @property
    def materials(self):
        return [matl_dict[self.key]]
