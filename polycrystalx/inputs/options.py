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


Output = namedtuple(
    "Output", ["fields"]
)
Output.__doc__ = """Output Options

PARAMETERS
----------
fields: sequence
   comma-separated list of output fields to write; fields will depend on
   process being run
"""
