"""Run time options"""
from collections import namedtuple


Options = namedtuple(
    "Options", ["output"]
)
Options.__doc__ = """Options

PARAMETERS
----------
output: Output instance
   output data
"""


class LinearElasticity:
    """Class for Linear Elasticity Options

    There are no parameters to initialize with. By instantiating this class
    you create an instance with all option values initialized to default
    values. Then you update as needed.
    """
    def __init__(self):
        self.output = self._make_output()

    @classmethod
    def _make_output(cls):
        _flds = [
            "write_mesh", "write_grain_ids", "write_displacement",
            "write_strain", "write_stress", "grain_averages"
        ]
        _dflts = len(_flds)  * [True]
        Output = namedtuple("Output", _flds, defaults=_dflts)
        Output.__doc__ = """Output Options

        PARAMETERS
        ----------
        "write_mesh", "write_grain_ids", "write_displacement",
        "write_strain", "write_stress": bool
            write the mesh, grain ID array, and/or dipslacement arrays

        "grain_averages":
            write grain volumes and grain-averaged values
        """
        return Output()
