"""Polycrystal Input Module"""

import numpy as np

from polycrystalx import inputs
from polycrystal.microstructure.single_crystal import (
    SingleCrystal as SingleCrystalMicro
)


def get_microstructure_input(key):
    """Return a named polycrystal input"""
    return PolycrystalInput(key).polycrystal_input


class PolycrystalInput:
    """Builds polycrystal input for polycrystalx

    Parameters:
    ----------
    key: hashable
       not used because it is an isotropic single grain (so far)

    """

    def __init__(self, key):
        self.key = key

    @property
    def polycrystal_input(self):
        return inputs.polycrystal.Polycrystal(
            name=self.name,
            polycrystal=self.microstructure
        )

    @property
    def name(self):
        return 'sx'

    @property
    def microstructure(self):
        return SingleCrystalMicro(np.identity(3))
