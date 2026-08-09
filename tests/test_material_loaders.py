"""Tests for polycrystalx.loaders.material"""
import pytest

from polycrystal.elasticity.single_crystal import SingleCrystal as ElasticSingleCrystal
from polycrystal.elasticity.moduli_tools.stiffness_matrix import MatrixComponentSystem
from polycrystal.heat_transfer.single_crystal import SingleCrystal as ThermalSingleCrystal

from polycrystalx import inputs
from polycrystalx.loaders.material import LinearElasticity, HeatTransfer


@pytest.fixture
def elastic_material():
    return ElasticSingleCrystal("isotropic", (200.0, 120.0), name="test-elastic")


@pytest.fixture
def thermal_material():
    return ThermalSingleCrystal("isotropic", [20.0], name="test-thermal")


@pytest.fixture
def elastic_input(elastic_material):
    return inputs.material.LinearElasticity(
        name="test-elastic-input",
        materials=[elastic_material],
    )


@pytest.fixture
def thermal_input(thermal_material):
    return inputs.material.HeatTransfer(
        name="test-thermal-input",
        materials=[thermal_material],
    )


class TestLinearElasticityMaterialLoader:

    def test_materials_stored(self, elastic_input, elastic_material):
        ldr = LinearElasticity(elastic_input)
        assert ldr.materials == [elastic_material]

    def test_mandel_system_applied(self, elastic_input):
        ldr = LinearElasticity(elastic_input)
        for m in ldr.materials:
            assert m.system is MatrixComponentSystem.MANDEL

    def test_multiple_phases(self):
        materials = [
            ElasticSingleCrystal("isotropic", (200.0, 120.0), name="phase-0"),
            ElasticSingleCrystal("cubic", (166.0, 119.0, 80.0), name="phase-1"),
        ]
        userinput = inputs.material.LinearElasticity(
            name="two-phase", materials=materials
        )
        ldr = LinearElasticity(userinput)
        assert len(ldr.materials) == 2
        for m in ldr.materials:
            assert m.system is MatrixComponentSystem.MANDEL


class TestHeatTransferMaterialLoader:

    def test_materials_stored(self, thermal_input, thermal_material):
        ldr = HeatTransfer(thermal_input)
        assert ldr.materials == [thermal_material]

    def test_rejects_non_thermal_material(self, elastic_material):
        bad_input = inputs.material.HeatTransfer(
            name="bad-input",
            materials=[elastic_material],
        )
        with pytest.raises(ValueError):
            HeatTransfer(bad_input)
