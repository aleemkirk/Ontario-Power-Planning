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


# ============================================================================
# HOURLY RESOLUTION OBJECTIVES
# ============================================================================


def total_cost_objective_hourly(model):
    """
    Calculate total system cost for hourly resolution model (NPV).

    CRITICAL DIFFERENCES from annual model:
    - Operating costs sum over representative days with weights
    - Includes ramp violation penalty for soft constraints
    - Capital and maintenance costs remain annual

    Includes:
    - Capital costs (CAPEX): x[t,i] × capex[i] / (1 + r)^t
    - Operating costs (OPEX): Σ_d Σ_h p_hourly[t,d,h,i] × weight[d] × opex[i] / (1 + r)^t
    - Maintenance costs: N[t,i] × maintenance[i] / (1 + r)^t
    - Ramp penalty: Σ_d Σ_h ramp_violation[t,d,h,i] × weight[d] × ramp_penalty / (1 + r)^t

    Representative day weights:
    - Each rep day represents a certain number of actual days in the year
    - Sum of weights = 365 or 366 (leap year)
    - Operating costs scaled by weight[d] to get annual totals

    Args:
        model: Pyomo model instance with hourly resolution

    Returns:
        Pyomo objective expression (total NPV cost in $)
    """
    # Calculate year index (0-based for discount factor)
    def year_index(t):
        return t - model.start_year

    # ========================================================================
    # ANNUAL COSTS (unchanged from annual model)
    # ========================================================================

    # Capital costs (NPV) - same as annual model
    # x[t,i] is MW of new capacity, capex is $/kW, so multiply by 1000
    capex_cost = sum(
        model.x[t, i] * model.capex[i] * 1000 * model.discount_factor[year_index(t)]
        for t in model.years
        for i in model.plant_types
    )

    # Maintenance costs (NPV) - $/MW/year, same as annual model
    maintenance_cost = sum(
        model.N[t, i] * model.maintenance[i] * model.discount_factor[year_index(t)]
        for t in model.years
        for i in model.plant_types
    )

    # ========================================================================
    # HOURLY COSTS (new for hourly model)
    # ========================================================================

    # Operating costs (NPV) - sum over representative days with weights
    # p_hourly[t,d,h,i] is MW power output in hour h of rep day d
    # opex[i] is $/MWh
    # weight[d] is number of actual days this rep day represents
    # Total = Σ_d Σ_h (p_hourly × opex × weight) gives annual MWh × $/MWh = annual $
    opex_cost = sum(
        model.p_hourly[t, d, h, i] * model.opex[i] * model.rep_day_weight[d]
        * model.discount_factor[year_index(t)]
        for t in model.years
        for d in model.rep_days
        for h in model.hours
        for i in model.plant_types
    )

    # Ramp violation penalty (NPV) - penalize soft constraint violations
    # ramp_violation[t,d,h,i] is MW of ramp constraint violation
    # ramp_penalty is $/MW (typically $1000/MW)
    # weight[d] scales to annual impact
    if hasattr(model, 'ramp_violation') and hasattr(model, 'ramp_penalty'):
        ramp_penalty_cost = sum(
            model.ramp_violation[t, d, h, i] * model.ramp_penalty * model.rep_day_weight[d]
            * model.discount_factor[year_index(t)]
            for t in model.years
            for d in model.rep_days
            for h in model.hours
            for i in model.plant_types
        )
    else:
        # Hard constraints - no violation variable
        ramp_penalty_cost = 0

    total_cost = capex_cost + opex_cost + maintenance_cost + ramp_penalty_cost

    return total_cost


def total_emissions_objective_hourly(model):
    """
    Calculate total carbon emissions for hourly resolution model.

    CRITICAL DIFFERENCE from annual model:
    - Sums over representative days with weights instead of annual p[t,i]

    Total emissions = Σ_t Σ_d Σ_h Σ_i p_hourly[t,d,h,i] × weight[d] × emission_factor[i]

    Representative day weights:
    - Each rep day d represents weight[d] actual days in the year
    - Multiplying by weight[d] gives annual emissions from that pattern
    - Sum over all rep days gives total annual emissions
    - Sum over all years gives planning horizon emissions

    Args:
        model: Pyomo model instance with hourly resolution

    Returns:
        Pyomo objective expression (total tons CO2)
    """
    total_emissions = sum(
        model.p_hourly[t, d, h, i] * model.rep_day_weight[d] * model.emissions[i]
        for t in model.years
        for d in model.rep_days
        for h in model.hours
        for i in model.plant_types
    )

    return total_emissions


def multi_objective_hourly(model, alpha: float = 0.5):
    """
    Weighted sum multi-objective function for hourly resolution model.

    Minimize: α × (cost/cost_max) + (1-α) × (emissions/emissions_max)

    where:
    - α = 0: minimize emissions only
    - α = 1: minimize cost only
    - 0 < α < 1: trade-off between objectives

    Uses hourly objective functions that sum over representative days with weights.

    Args:
        model: Pyomo model instance with hourly resolution
        alpha: Weight for cost objective (0 to 1)

    Returns:
        Pyomo objective expression
    """
    # Get individual hourly objectives
    cost = total_cost_objective_hourly(model)
    emissions = total_emissions_objective_hourly(model)

    # Normalize using reference values stored in model
    # These should be set before calling this function by running
    # cost-only and emissions-only optimizations
    if hasattr(model, 'cost_normalization') and hasattr(model, 'emissions_normalization'):
        cost_normalized = cost / model.cost_normalization
        emissions_normalized = emissions / model.emissions_normalization
        return alpha * cost_normalized + (1 - alpha) * emissions_normalized
    else:
        # No normalization available - use raw weighted sum
        # This works but may give poor scaling if objectives have very different magnitudes
        return alpha * cost + (1 - alpha) * emissions
