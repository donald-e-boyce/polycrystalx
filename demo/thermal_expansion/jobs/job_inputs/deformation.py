"""Deformation Input Module"""

import numpy as np

from polycrystalx import inputs
from polycrystalx.inputs.tools import interpolate


zero = inputs.function.Function(
    source="constant",
    value=(0, 0, 0),
)


def get_deformation_input(key, matl_inp):
    """Return a named deformation input"""
    return DeformationInput(key, matl_inp).deformation_input


class DeformationInput:
    """Builds deformation input for deformation

    Parameters:
    ----------
    key: str
       key can be one of three strings:
        "zero": constant thermal strain and zero displacement boundary conditions
        "match": constant thermal strain field and boundary conditions that match
        "linear": linear thermal strain field and zero boundary conditions
    matl_inp: inputs.MaterialInput
        the material input for this job
    """

    def __init__(self, key, matl_inp):
        self.key = key
        self.matl_inp = matl_inp
        self.matl = matl_inp.materials[0]

        # Note that matl.cte() is a function of reference temperature.
        self.linear_cte = self.matl.cte(0)

        # This sets up the displacement bcs and thermal expansion.
        self._setbcs()

    @property
    def deformation_input(self):
        return inputs.deformation.LinearElasticity(
            name = self.name,
            force_density = self.body_force,
            displacement_bcs = self.displacement_bcs,
            traction_bcs = self.traction_bcs,
            thermal_expansion = self.thermal_expansion,
        )

    @property
    def name(self):
        return self.key

    @property
    def body_force(self):
        return inputs.function.Function(
            source="constant",
            value=(0, 0, 0),
        )

    @property
    def displacement_bcs(self):
        return self._displacement_bcs

    @property
    def traction_bcs(self):
        return []

    @property
    def thermal_expansion(self):
        return self._thermal_expansion

    def linear_strain(self, x):
        """Linear thermal strain field"""
        fac = 200 + x[0, :] * 50
        tmp = fac.reshape(-1, 1) * self.linear_cte.flatten()
        return tmp.T

    def _setbcs(self):
        constant_diff = 300 * self.linear_cte

        # Possible thermal strain fields.

        constant_expansion = inputs.function.Function(
            source="constant",
            value=constant_diff.flatten(),
        )

        linear_expansion = inputs.function.Function(
            source="interpolation",
            function=self.linear_strain,
        )

        # Possible boundary conditions.
        zero_bcs = [
            inputs.deformation.DisplacementBC(
                section = "boundary",
                value = interpolate.linear_function(np.zeros((3, 3))),
            )
        ]

        match_bcs = [
            inputs.deformation.DisplacementBC(
                section = "boundary",
                value = interpolate.linear_function(constant_diff),
            )
        ]

        # Now, set the fields according to key.

        if self.key == "zero":
            # Constant expansion, zero boundary conditions

            self._thermal_expansion = constant_expansion
            self._displacement_bcs = zero_bcs

        elif self.key == "match":
            # Constant expansion, boundary conditions match expansion

            self._thermal_expansion= constant_expansion
            self._displacement_bcs = match_bcs

        elif self.key == "linear":
            # Linear expansion, zero boundary condtions

            self._thermal_expansion = linear_expansion
            self._displacement_bcs = zero_bcs
