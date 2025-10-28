"""
Decision variable definitions for the optimization model.

Defines:
- x[t,i]: Number/capacity of plants of type i to build in year t
- p[t,i,h]: Power output from plant type i in year t, hour h
- N[t,i]: Total operating capacity of plant type i in year t
"""

import pyomo.environ as pyo


def define_variables(model):
    """
    Define all decision variables for the optimization model.

    Args:
        model: Pyomo ConcreteModel instance
    """
    # TODO: Implement variable definitions
    pass
