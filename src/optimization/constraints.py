"""
Shared constraint definitions for hourly optimization model.

Contains:
- Reserve margin constraints
- Capacity dynamics (construction lead times, retirements)
"""

import pyomo.environ as pyo


def reserve_margin_constraint(model):
    """
    Maintain 15% reserve margin above peak demand.

    Σ_i N[t,i] ≥ (1 + ReserveMargin) × PeakDemand[t]  ∀t

    Args:
        model: Pyomo model instance
    """
    def reserve_rule(m, t):
        total_capacity = sum(m.N[t, i] for i in m.plant_types)
        required_capacity = (1 + m.reserve_margin) * m.peak_demand[t]
        return total_capacity >= required_capacity

    model.reserve_margin_constraint = pyo.Constraint(model.years, rule=reserve_rule)


def capacity_dynamics_constraint(model):
    """
    Track capacity evolution over time.

    Full version (Phase 3) with lead times and retirements:
    - N[t,i] = N[t-1,i] + NewCapacity[t,i] - Retirements[t,i]  ∀t,i
    - NewCapacity[t,i] = x[t-lead_time[i],i] (plants built lead_time years ago)
    - Retirements tracked separately based on plant age

    Args:
        model: Pyomo model instance
    """
    # Check if model has lead time and retirement tracking enabled
    use_lead_times = hasattr(model, 'lead_time')
    use_retirements = hasattr(model, 'retirement')

    def initial_capacity_rule(m, i):
        """Set initial capacity for first year."""
        return m.N[m.start_year, i] == m.initial_capacity[i]

    def capacity_evolution_rule(m, t, i):
        """Track capacity evolution year-by-year with lead times and retirements."""
        if t == m.start_year:
            return pyo.Constraint.Skip

        prev_year = t - 1

        # Calculate new capacity coming online (accounting for lead time)
        if use_lead_times:
            lead_time = int(pyo.value(m.lead_time[i]))
            build_year = t - lead_time

            # Only count builds from years within our planning horizon
            if build_year >= m.start_year and build_year in m.years:
                new_capacity = m.x[build_year, i]
            else:
                new_capacity = 0
        else:
            # No lead time - plants available immediately
            new_capacity = m.x[t, i]

        # Calculate retirements
        if use_retirements:
            retirements = m.retirement[t, i]
        else:
            retirements = 0

        return m.N[t, i] == m.N[prev_year, i] + new_capacity - retirements

    model.initial_capacity_constraint = pyo.Constraint(
        model.plant_types, rule=initial_capacity_rule
    )

    model.capacity_evolution_constraint = pyo.Constraint(
        model.years, model.plant_types, rule=capacity_evolution_rule
    )
