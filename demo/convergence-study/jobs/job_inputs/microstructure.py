"""Polycrystal Input Module

Keys
----
(vfp0, misori_deg)       : bicrystal (volume fraction %, misorientation degrees)
("voronoi", ng, msid)    : random Voronoi loaded from jobs/data/micro-{ng}g-{msid}.npz
"""

import os

import numpy as np

from polycrystalx import inputs
from polycrystal.microstructure.analytic import Analytic
from polycrystal.microstructure.voronoi import Voronoi

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def get_polycrystal_input(key):
    """Return a polycrystal input for the given key.

    Parameters
    ----------
    key: tuple
        Either (vfp0, misori_deg) for a bicrystal, or
        ("voronoi", ng, msid) for a random Voronoi microstructure.
    """
    if key[0] == "voronoi":
        return _VoronoiInput(key).polycrystal_input
    else:
        return _BicrystalInput(key).polycrystal_input


class _BicrystalInput:
    """Bicrystal microstructure input.

    Parameters
    ----------
    key: tuple -> (vfp0, misori_deg)
        volume fraction percentage and misorientation angle in degrees
    """

    def __init__(self, key):
        self.key = key
        self.vfp0, self.misori_deg = key

    @property
    def polycrystal_input(self):
        return inputs.polycrystal.Polycrystal(
            name=self.name,
            polycrystal=self.microstructure
        )

    @property
    def name(self):
        return f"bicrystal-{self.vfp0}pct-{self.misori_deg}mis"

    @property
    def ori_list(self):
        ori_0 = np.identity(3)
        ori_1 = np.identity(3)
        a = np.radians(self.misori_deg)
        ori_1[1, 1:] = [np.cos(a), np.sin(a)]
        ori_1[2, 1:] = [-np.sin(a), np.cos(a)]
        return np.stack([ori_0, ori_1])

    def grain_fun(self, x):
        x0 = x[:, 0]
        return np.where(x0 <= self.vfp0 * 0.01, 0, 1)

    @property
    def microstructure(self):
        return Analytic(self.grain_fun, self.ori_list)


class _VoronoiInput:
    """Random Voronoi microstructure input.

    Parameters
    ----------
    key: tuple -> ("voronoi", ng, msid)
        ng   — number of grains
        msid — microstructure sequence number (matches gen_voronoi.py numbering)
    """

    def __init__(self, key):
        self.key = key
        _, self.ng, self.msid = key

    @property
    def polycrystal_input(self):
        return inputs.polycrystal.Polycrystal(
            name=self.name,
            polycrystal=self.microstructure
        )

    @property
    def name(self):
        return f"voronoi-{self.ng}g-{self.msid}"

    @property
    def filepath(self):
        return os.path.join(_DATA_DIR, f"micro-{self.ng}g-{self.msid}.npz")

    @property
    def microstructure(self):
        return Voronoi.from_file(self.filepath)
