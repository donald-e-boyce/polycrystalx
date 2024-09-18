"""Tests for inputs"""
import numpy as np

from polycrystalx import inputs


def test_mesh_inputs():

    mesh_input = inputs.mesh.Mesh(
        name='test',
        source="box",
    )

    assert mesh_input.extents is None
    assert mesh_input.divisions is None
    assert mesh_input.celltype is None
    assert mesh_input.file is None
    assert mesh_input.boundary_sections == []


def test_deformation_inputs():
    defm_input = inputs.deformation.LinearElasticity(
        name=("test"),
    )
    assert defm_input.force_density is None
    assert defm_input.plastic_distortion is None
    assert defm_input.displacement_bcs == []
    assert defm_input.traction_bcs == []


def test_options():

    options = inputs.options.LinearElasticity()

    assert options.output.write_mesh == True
    assert options.output.write_grain_ids == True
    assert options.output.write_displacement == True
    assert options.output.write_strain == True
    assert options.output.write_stress == True
    assert options.output.grain_averages == True

    options.output = options.output._replace(
        write_mesh=False,
        write_grain_ids=False,
        write_displacement=False,
        write_strain=False,
        write_stress=False,
        grain_averages=False,
    )
    assert options.output.write_mesh == False
    assert options.output.write_grain_ids == False
    assert options.output.write_displacement == False
    assert options.output.write_strain == False
    assert options.output.write_stress == False
    assert options.output.grain_averages == False
