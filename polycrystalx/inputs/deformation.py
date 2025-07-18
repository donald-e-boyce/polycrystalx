"""Deformation input templates"""
from collections import namedtuple

import numpy as np

from .function import Function


def boundary_values_template(x):
    """Boundary values function template

    Parameters
    ----------
    x: numpy array of shape (d, n)
       array of `n` points of dimension `d`

    Returns
    -------
    array
       scalar of length `n` or vector of shape `(d, n)`
    """
    return np.zeros(x.shape)


BoundaryCondition = namedtuple(
    "BoundaryCondition", ["section", "value", "component"],
    defaults = [None]
)
BoundaryCondition.__doc__ = """Boundary condition

Set up boundary conditions on the named surface with a function that returns
the boundary values. The boundary conditions are vector-valued unless a
component is specified, in which they are scalar valued for that component.

Parameters
-----------
section: str
    the name of the boundary section
value: function (see below)
    function of position giving boundary values
component: int, optional
    if specified, index of component with applied BC

See Also
--------
boundary_values_template: template for boundary value functions
"""


# This section is for the LinearElasticity process.

DirichletBC = BoundaryCondition
DisplacementBC = BoundaryCondition
TractionBC = BoundaryCondition


LinearElasticity = namedtuple(
    "LinearElasticity",
    ["name", "force_density", "plastic_distortion", "thermal_expansion",
     "displacement_bcs", "traction_bcs"],
    defaults=[None, None, None, [], []]
)
LinearElasticity.__doc__ = """Deformation input for Elasticity

Parameters
-----------
name: str
    name of this deformation input
force_density: inputs.function.Function
    force density function specification
plastic_distortion: inputs.function.Function
    plastic distortion function specification
thermal_expansion: inputs.function.Function
    thermal expansion tensor function
displacement_bcs: list
    list of DisplacementBC instances
traction_bcs: list of inputs.deformation.TractionBC
    list of traction boundary condition specifications
"""


# This section is for the HeatTransfer process.

TemperatureBC = BoundaryCondition
FluxBC = BoundaryCondition


HeatTransfer = namedtuple(
    "HeatTransfer", ["name", "body_heat", "temperature_bcs", "flux_bcs"],
    defaults=[None, [], []]
)
HeatTransfer.__doc__ = """Deformation input for heat transfer

Parameters
-----------
name: str
    name of this deformation input
body_heat: inputs.function.Function
    force density function specification
temperature_bcs: list
    list of temperature boundary condition specifications
flux_bcs: list of inputs.deformation.TractionBC
    list of flux boundary condition specifications
"""
