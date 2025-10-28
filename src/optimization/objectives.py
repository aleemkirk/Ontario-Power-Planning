"""
Objective function definitions.

Contains:
- Total system cost (capital + operating + maintenance)
- Total carbon emissions
- Multi-objective weighted sum
"""

import pyomo.environ as pyo


def total_cost_objective(model):
    """
    Calculate total system cost (NPV).

    Includes:
    - Capital costs (CAPEX): x[t,i] × capex[i] / (1 + r)^t
    - Operating costs (OPEX): p[t,i] × opex[i] / (1 + r)^t
    - Maintenance costs: N[t,i] × maintenance[i] / (1 + r)^t

    Args:
        model: Pyomo model instance

    Returns:
        Pyomo objective expression (total NPV cost in $)
    """
    # Calculate year index (0-based for discount factor)
    def year_index(t):
        return t - model.start_year

    # Capital costs (NPV)
    capex_cost = sum(
        model.x[t, i] * model.capex[i] * 1000 * model.discount_factor[year_index(t)]
        for t in model.years
        for i in model.plant_types
    )

    # Operating costs (NPV) - convert $/MWh to total cost
    opex_cost = sum(
        model.p[t, i] * model.opex[i] * model.discount_factor[year_index(t)]
        for t in model.years
        for i in model.plant_types
    )

    # Maintenance costs (NPV) - $/MW/year
    maintenance_cost = sum(
        model.N[t, i] * model.maintenance[i] * model.discount_factor[year_index(t)]
        for t in model.years
        for i in model.plant_types
    )

    return capex_cost + opex_cost + maintenance_cost


def total_emissions_objective(model):
    """
    Calculate total carbon emissions over planning horizon.

    Total emissions = Σ_t Σ_i p[t,i] × emission_factor[i]

    Args:
        model: Pyomo model instance

    Returns:
        Pyomo objective expression (total tons CO2)
    """
    total_emissions = sum(
        model.p[t, i] * model.emissions[i]
        for t in model.years
        for i in model.plant_types
    )

    return total_emissions


def multi_objective(model, alpha: float = 0.5):
    """
    Weighted sum multi-objective function.

    Minimize: α × (cost/cost_max) + (1-α) × (emissions/emissions_max)

    where:
    - α = 0: minimize emissions only
    - α = 1: minimize cost only
    - 0 < α < 1: trade-off between objectives

    Args:
        model: Pyomo model instance
        alpha: Weight for cost objective (0 to 1)

    Returns:
        Pyomo objective expression
    """
    # Get individual objectives
    cost = total_cost_objective(model)
    emissions = total_emissions_objective(model)

    # Normalize using reference values stored in model
    # These should be set before calling this function
    cost_normalized = cost / model.cost_normalization
    emissions_normalized = emissions / model.emissions_normalization

    return alpha * cost_normalized + (1 - alpha) * emissions_normalized
