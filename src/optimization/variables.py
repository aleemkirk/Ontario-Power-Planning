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

    Variables created:
        x[t,i]: New capacity of plant type i to build in year t (MW) - continuous, non-negative
        p[t,i]: Annual energy generation from plant type i in year t (MWh) - continuous, non-negative
        N[t,i]: Total operating capacity of plant type i in year t (MW) - continuous, non-negative
    """
    # x[t,i] - New capacity to build (MW)
    model.x = pyo.Var(
        model.years,
        model.plant_types,
        domain=pyo.NonNegativeReals,
        doc="New capacity to build (MW)"
    )

    # p[t,i] - Annual energy generation (MWh)
    model.p = pyo.Var(
        model.years,
        model.plant_types,
        domain=pyo.NonNegativeReals,
        doc="Annual energy generation (MWh)"
    )

    # N[t,i] - Total operating capacity (MW)
    model.N = pyo.Var(
        model.years,
        model.plant_types,
        domain=pyo.NonNegativeReals,
        doc="Total operating capacity (MW)"
    )
