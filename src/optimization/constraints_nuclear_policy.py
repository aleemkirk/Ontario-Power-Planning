"""
Nuclear policy constraint: Ensure nuclear accounts for at least 50% of generation.

This constraint ensures that nuclear energy accounts for at least 50% of all power
generated in any given year, representing a policy commitment to nuclear baseload.
"""

import pyomo.environ as pyo


def nuclear_minimum_share_constraint(model, min_share=0.5):
    """
    Ensure nuclear energy accounts for at least 50% of generation each year.

    Constraint: Σ_d Σ_h p_hourly[t,d,h,'nuclear'] × weight[d] >= 0.5 × Σ_d Σ_h Σ_i p_hourly[t,d,h,i] × weight[d]

    This can be rewritten as:
    Σ_d Σ_h (p_hourly[t,d,h,'nuclear'] - 0.5 × Σ_i p_hourly[t,d,h,i]) × weight[d] >= 0

    Args:
        model: Pyomo model instance with hourly resolution
        min_share: Minimum share of nuclear generation (default 0.5 = 50%)

    Returns:
        None (modifies model in-place)
    """
    def nuclear_share_rule(m, t):
        """Ensure nuclear share >= min_share for each year."""
        # Nuclear generation (weighted by rep days)
        nuclear_gen = sum(
            m.p_hourly[t, d, h, 'nuclear'] * m.rep_day_weight[d]
            for d in m.rep_days
            for h in m.hours
        )

        # Total generation (weighted by rep days)
        total_gen = sum(
            m.p_hourly[t, d, h, i] * m.rep_day_weight[d]
            for d in m.rep_days
            for h in m.hours
            for i in m.plant_types
        )

        # Nuclear generation must be at least min_share of total
        return nuclear_gen >= min_share * total_gen

    model.nuclear_minimum_share = pyo.Constraint(
        model.years,
        rule=nuclear_share_rule,
        doc=f"Nuclear generation >= {min_share*100:.0f}% of total generation"
    )

    print(f"✓ Added nuclear minimum share constraint: {min_share*100:.0f}% of generation")
    print(f"  Constraints added: {len(model.nuclear_minimum_share):,}")


def add_nuclear_policy_constraints(model, min_nuclear_share=0.5):
    """
    Add all nuclear policy constraints to the model.

    Args:
        model: Pyomo model instance with hourly resolution
        min_nuclear_share: Minimum share of nuclear generation (default 0.5 = 50%)
    """
    print("\n[Adding Nuclear Policy Constraints]")
    nuclear_minimum_share_constraint(model, min_share=min_nuclear_share)
    print("✓ Nuclear policy constraints added successfully")
