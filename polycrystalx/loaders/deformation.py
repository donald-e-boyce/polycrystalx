"""Deformation input templates"""
import numpy as np
from dolfinx import fem, mesh
import ufl

from ..forms.linear_elasticity import Traction
from ..forms.heat_transfer import Flux
from .function import FunctionLoader


class DefmLoader:
    """Base class for deformation loaders

    Parameters
    ----------
    job: inputs.Job instance
       input specification
    """

    def __init__(self, defm_input):
        self.defm_input = defm_input

    @staticmethod
    def boundary_measures(bcs, V, bdict):
        """Return boundary measures from list of bcs"""
        bdim = V.mesh.topology.dim - 1
        flist, vlist = [], []
        for i, bc in enumerate(bcs):
            facets = bdict[bc.section]
            flist.append(facets)
            vlist.append((i + 1) * np.ones(len(facets), dtype=np.int32))
        mtags = mesh.meshtags(
            V.mesh, bdim, np.hstack(flist), np.hstack(vlist)
        )
        return ufl.Measure("ds", subdomain_data=mtags)


class LinearElasticity(DefmLoader):

    def displacement_bcs(self, V, bdict):
        """Return list of Dirichlet BCs for this problem

        Parameters
        ----------
        V: dolfinx FunctionSpace
           the vector function space
        bdict: dict
           the boundary dictionary

        Returns
        -------
        list
           list of displacement (Dirichlet) boundary conditions
        """
        bdim = V.mesh.topology.dim - 1
        dbcs = []
        for dbc in self.defm_input.displacement_bcs:
            facets = bdict[dbc.section]
            if dbc.component is None:
                dofs = fem.locate_dofs_topological(
                    V=V, entity_dim=bdim, entities=facets
                )
                ubc = fem.Function(V)
                ubc.interpolate(dbc.value)
                bc = fem.dirichletbc(value=ubc, dofs=dofs)
            else:
                Vbc = V.sub(dbc.component)
                Vc, _dofmap = Vbc.collapse()
                dofs_vector = fem.locate_dofs_topological(
                    V=V.sub(dbc.component), entity_dim=bdim, entities=facets
                )
                dofs_scalar = fem.locate_dofs_topological(
                    V=Vc, entity_dim=bdim, entities=facets
                )
                dofs = [dofs_vector, dofs_scalar]
                ubc = fem.Function(Vc)
                ubc.interpolate(dbc.value)
                bc = fem.dirichletbc(value=ubc, dofs=dofs, V=Vbc)
            dbcs.append(bc)

        return dbcs


    def traction_bcs(self, V, bdict):
        """Return list of traction BCs for this problem

        Parameters
        ----------
        V: dolfinx FunctionSpace
           the vector function space
        bdict: dict
           the boundary dictionary

        Returns
        -------
        list
           list of traction (natural) boundary conditions
        """
        if len(self.defm_input.traction_bcs) == 0:
            return []
        #
        # First, create the surface measure subdomain data using defined by
        # meshtags.
        #
        ds = self.boundary_measures(self.defm_input.traction_bcs, V, bdict)
        #
        # Next, create the array of traction forms.
        #
        tbcs = []
        for i, tbc in enumerate(self.defm_input.traction_bcs):
            Vbc = V if tbc.component is None else V.sub(tbc.component)
            ubc = fem.Function(Vbc)
            ubc.interpolate(tbc.value)
            t = Traction(ubc, ds(i + 1), tbc.component)
            tbcs.append(t)

        return tbcs


    def force_density(self, V):
        """Return force density function

        Parameters
        ----------
        V: dolfinx FunctionSpace
           vector function space for force density

        Returns
        -------
        dolfinx Function
           body force function as specified
        """
        if self.defm_input.force_density is not None:
            return FunctionLoader(self.defm_input.force_density).load(V)

    def plastic_distortion(self, T):
        """Return plastic distortion function

        Parameters
        ----------
        T: dolfinx FunctionSpace
           tensor function space for plastic distortion

        Returns
        -------
        dolfinx Function
           plastic distortion function as specified
        """
        if self.defm_input.plastic_distortion is not None:
            return FunctionLoader(self.defm_input.plastic_distortion).load(T)

    def thermal_expansion(self, T):
        """Return thermal expansion function

        Parameters
        ----------
        T: dolfinx FunctionSpace
           tensor function space for thermal expansion

        Returns
        -------
        dolfinx Function
           thermal expansion function as specified
        """
        if self.defm_input.thermal_expansion is not None:
            return FunctionLoader(self.defm_input.thermal_expansion).load(T)


class HeatTransfer(DefmLoader):
    """Loader for heat transfer inputs"""

    def body_heat(self, V):
        """Return body heat density function

        Parameters
        ----------
        V: dolfinx FunctionSpace
           vector function space for body heat density

        Returns
        -------
        dolfinx Function
           body heat function as specified
        """
        if self.defm_input.body_heat is not None:
            return FunctionLoader(self.defm_input.body_heat).load(V)

    def temperature_bcs(self, V, bdict):
        """Return list of Dirichlet BCs for this problem

        Parameters
        ----------
        V: dolfinx FunctionSpace
           the vector function space
        bdict: dict
           the boundary dictionary

        Returns
        -------
        list
           list of temperature (Dirichlet) boundary conditions
        """
        bdim = V.mesh.topology.dim - 1
        dbcs = []
        for dbc in self.defm_input.temperature_bcs:
            facets = bdict[dbc.section]
            dofs = fem.locate_dofs_topological(
                V=V, entity_dim=bdim, entities=facets
            )
            ubc = fem.Function(V)
            ubc.interpolate(dbc.value)
            bc = fem.dirichletbc(value=ubc, dofs=dofs)
            dbcs.append(bc)

        return dbcs

    def flux_bcs(self, V, bdict):
        """Return list of flux BCs for this problem

        Parameters
        ----------
        V: dolfinx FunctionSpace
           the function space
        bdict: dict
           the boundary dictionary

        Returns
        -------
        list
           list of flux (natural) boundary conditions
        """
        if len(self.defm_input.flux_bcs) == 0:
            return []
        #
        # First, create the surface measure subdomain data using defined by
        # meshtags.
        #
        ds = self.boundary_measures(self.defm_input.flux_bcs, V, bdict)
        #
        # Next, create the array of traction forms.
        #
        fbcs = []
        for i, fbc in enumerate(self.defm_input.flux_bcs):
            ubc = fem.Function(V)
            ubc.interpolate(fbc.value)
            f = Flux(ubc, ds(i + 1))
            fbcs.append(f)

        return fbcs
