"""
Hourly constraint definitions for the optimization model.

Contains:
- Hourly demand satisfaction constraints
- Hourly capacity constraints
- Ramp rate constraints (ACTIVATED for hourly model)
"""

import pyomo.environ as pyo


def hourly_demand_satisfaction_constraint(model):
    """
    Ensure total generation meets demand every hour of every representative day.

    Constraint: sum(p_hourly[t,d,h,i]) >= hourly_demand[t,d,h]  ∀t,d,h

    Args:
        model: Pyomo model instance
    """
    def demand_rule(m, t, d, h):
        return sum(m.p_hourly[t, d, h, i] for i in m.plant_types) >= m.hourly_demand[t, d, h]

    model.hourly_demand_constraint = pyo.Constraint(
        model.years, model.rep_days, model.hours,
        rule=demand_rule,
        doc="Hourly demand satisfaction"
    )

    print(f"✓ Added hourly demand constraints: {len(model.hourly_demand_constraint):,}")


def hourly_capacity_constraint(model):
    """
    Ensure hourly generation doesn't exceed available capacity.

    Constraint: p_hourly[t,d,h,i] <= N[t,i] × CF[i]  ∀t,d,h,i

    Note: No multiplication by hours (8760) as in annual model - this is power (MW), not energy (MWh)

    Args:
        model: Pyomo model instance
    """
    def capacity_rule(m, t, d, h, i):
        return m.p_hourly[t, d, h, i] <= m.N[t, i] * m.capacity_factor[i]

    model.hourly_capacity_constraint = pyo.Constraint(
        model.years, model.rep_days, model.hours, model.plant_types,
        rule=capacity_rule,
        doc="Hourly capacity limits"
    )

    print(f"✓ Added hourly capacity constraints: {len(model.hourly_capacity_constraint):,}")


def ramp_rate_constraint_hourly(model, use_soft=True):
    """
    Limit rate of change in power output between consecutive hours.

    CRITICAL: This constraint was DISABLED in the annual model.
    Now ACTIVATED for hourly resolution!

    Hard constraint: |p[t,d,h,i] - p[t,d,h-1,i]| <= RampRate[i] × N[t,i] × 60
    Soft constraint: Same but with slack variable ramp_violation[t,d,h,i]

    RampRate[i] is in MW/min per MW of capacity
    Multiply by 60 to convert to MW/hour

    Example ramp rates:
    - Hydro: 0.15 MW/min → 9 MW/hour per MW capacity
    - Gas: 0.04 MW/min → 2.4 MW/hour per MW capacity
    - Nuclear: 0.02 MW/min → 1.2 MW/hour per MW capacity

    Args:
        model: Pyomo model instance
        use_soft: If True, use soft constraints with penalty (recommended)
    """
    if use_soft:
        # Soft constraint: allow violations but penalize them

        def ramp_up_rule(m, t, d, h, i):
            if h == 0:
                # Skip first hour of each day (no previous hour to compare)
                return pyo.Constraint.Skip

            # Maximum ramp rate: MW/min × MW capacity × 60 min/hour = MW/hour
            max_ramp_mw_per_hour = m.ramp_rate[i] * m.N[t, i] * 60

            # Allow violation via slack variable
            return (m.p_hourly[t, d, h, i] - m.p_hourly[t, d, h-1, i]
                    <= max_ramp_mw_per_hour + m.ramp_violation[t, d, h, i])

        def ramp_down_rule(m, t, d, h, i):
            if h == 0:
                return pyo.Constraint.Skip

            max_ramp_mw_per_hour = m.ramp_rate[i] * m.N[t, i] * 60

            # Ramp down: previous hour - current hour
            return (m.p_hourly[t, d, h-1, i] - m.p_hourly[t, d, h, i]
                    <= max_ramp_mw_per_hour + m.ramp_violation[t, d, h, i])

    else:
        # Hard constraint: absolutely enforce ramp limits

        def ramp_up_rule(m, t, d, h, i):
            if h == 0:
                return pyo.Constraint.Skip

            max_ramp_mw_per_hour = m.ramp_rate[i] * m.N[t, i] * 60
            return m.p_hourly[t, d, h, i] - m.p_hourly[t, d, h-1, i] <= max_ramp_mw_per_hour

        def ramp_down_rule(m, t, d, h, i):
            if h == 0:
                return pyo.Constraint.Skip

            max_ramp_mw_per_hour = m.ramp_rate[i] * m.N[t, i] * 60
            return m.p_hourly[t, d, h-1, i] - m.p_hourly[t, d, h, i] <= max_ramp_mw_per_hour

    model.ramp_up_constraint = pyo.Constraint(
        model.years, model.rep_days, model.hours, model.plant_types,
        rule=ramp_up_rule,
        doc="Ramp up rate limit"
    )

    model.ramp_down_constraint = pyo.Constraint(
        model.years, model.rep_days, model.hours, model.plant_types,
        rule=ramp_down_rule,
        doc="Ramp down rate limit"
    )

    constraint_type = "soft (with penalty)" if use_soft else "hard (strict)"
    print(f"✓ Added ramp rate constraints ({constraint_type}): {len(model.ramp_up_constraint) + len(model.ramp_down_constraint):,}")
    print(f"  🔥 RAMP RATE CONSTRAINTS NOW ACTIVE!")


def add_hourly_constraints(model, use_soft_ramp=True):
    """
    Add all hourly-specific constraints to the model.

    Args:
        model: Pyomo model instance
        use_soft_ramp: Use soft ramp constraints with penalty (default True)
    """
    print("\n[Adding Hourly Constraints]")

    # Hourly demand satisfaction
    hourly_demand_satisfaction_constraint(model)

    # Hourly capacity limits
    hourly_capacity_constraint(model)

    # Ramp rate constraints (ACTIVATED!)
    ramp_rate_constraint_hourly(model, use_soft=use_soft_ramp)

    # Calculate total hourly constraints
    total_hourly = (len(model.hourly_demand_constraint) +
                   len(model.hourly_capacity_constraint) +
                   len(model.ramp_up_constraint) +
                   len(model.ramp_down_constraint))

    print(f"\n✓ Total hourly constraints added: {total_hourly:,}")
