"""Material Input Module

References
----------
* Ti-64 data from matweb:
  Titanium Ti-6Al-4V (Grade 5), Annealed Bar
  https://www.matweb.com/search/DataSheet.aspx?MatGUID=10d463eb3d3d4ff48fc57e0ad1037434
"""

from polycrystalx import inputs

from polycrystal.elasticity.single_crystal import SingleCrystal


""" CTE Data from matweb
20 - 100 C: 8.6 u/K
20 - 315 C: 9.2 u/K
20 - 650 C: 9.7 u/K
"""
E, nu = 114.0, 0.33
cte_RT = 8.6e-6

matl_dict = {
    "identity-iso": SingleCrystal.from_K_G(1/3, 1/2, cte=cte_RT),
    "ti-64-bar-RT": SingleCrystal.from_E_nu(E, nu, units="GPa", cte=cte_RT),
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
        self.name = key

    @property
    def material_input(self):
        return inputs.material.LinearElasticity(
            name=self.name,
            materials=self.materials
        )

    @property
    def materials(self):
        return [matl_dict[self.name]]
