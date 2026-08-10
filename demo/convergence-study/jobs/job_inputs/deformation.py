"""Deformation Input Module"""

import numpy as np

from polycrystalx import inputs
from polycrystalx.inputs.tools import interpolate


def get_deformation_input(key):
    """Return a named deformation input"""
    return DeformationInput.registry[key]().deformation_input


class DeformationInput:
    """This is the base class for building deformation inputs

    Each subclass will have a name and be stored in registry. So the deformation
    key is the class name.
    """
    registry = dict()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "name"):
            cls.registry[cls.name] = cls

    @property
    def deformation_input(self):
        return inputs.deformation.HeatTransfer(
            name=self.name,
            body_heat=self.body_heat,
            temperature_bcs=self.temperature_bcs,
            flux_bcs=self.flux_bcs
        )

    @property
    def body_heat(self):
        return inputs.function.Function(
            source="constant",
            value=0.0,
        )

    @property
    def flux_bcs(self):
        return []

    @property
    def temperature_bcs(self):
        return []


class TempZ(DeformationInput):
    """First, simple test with temperatures preescribed at top and bottom z-surfaces"""

    name = "temp-z"

    @property
    def temperature_bcs(self):
        return [
            inputs.deformation.TemperatureBC(
                    section = "zmin",
                    value = interpolate.constant(0.0),
                ),

            inputs.deformation.TemperatureBC(
                    section = "zmax",
                    value = interpolate.constant(1.0),
                ),

        ]


class TempZ_hs(TempZ):
    """Same boundary conditions as TempZ, but adding a heat source"""

    name = "temp-z-hs"

    @staticmethod
    def heat_func(x):
        n = x.shape[1]
        cen = (0.3, 0.4, 0.65)
        d2 = np.zeros(n)
        for i in range(3):
            d2 += (x[i, :] - cen[i]) ** 2
        return 10.0 / (1.0 + d2)

    @property
    def body_heat(self):
        return inputs.function.Function(
            source="interpolation",
            function=self.heat_func,
        )
