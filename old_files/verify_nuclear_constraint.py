"""
Verify that the 50% nuclear generation constraint is actually satisfied.

This script runs the optimization with nuclear policy and checks year-by-year
whether nuclear actually accounts for ≥50% of generation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.optimization.model_hourly import PowerSystemOptimizationHourly
import pandas as pd

print("="*80)
print("VERIFYING 50% NUCLEAR GENERATION CONSTRAINT")
print("="*80)

# Run emissions-optimal case with nuclear policy
print("\nRunning emissions-optimal (α=0.0) WITH 50% nuclear policy...")
optimizer = PowerSystemOptimizationHourly(
    start_year=2025,
    end_year=2045,
    n_rep_days=12,
    use_soft_ramp_constraints=True,
    ramp_penalty=1000.0,
    use_lead_times=True,
    use_retirements=False,
    use_nuclear_policy=True,
    min_nuclear_share=0.5
)

optimizer.build_model(objective='emissions')
result = optimizer.solve(solver='highs', time_limit=600, tee=False, save_results=False)

# Access the Pyomo model directly
m = optimizer.model

print("\n" + "="*80)
print("NUCLEAR GENERATION SHARE BY YEAR")
print("="*80)
print(f"{'Year':<8} {'Nuclear (GWh)':<18} {'Total (GWh)':<18} {'Nuclear %':<12} {'≥50%?'}")
print("-"*80)

import pyomo.environ as pyo

all_satisfied = True

for year in range(2025, 2046):
    # Calculate nuclear generation for this year (across all rep days and hours)
    nuclear_gen = 0
    total_gen = 0

    for d in m.rep_days:
        for h in m.hours:
            # Nuclear generation in this hour
            try:
                p_nuc = pyo.value(m.p_hourly[year, d, h, 'nuclear'])
                if p_nuc is None:
                    p_nuc = 0.0
            except:
                p_nuc = 0.0

            weight = pyo.value(m.rep_day_weight[d])
            nuclear_gen += p_nuc * weight

            # Total generation in this hour
            for plant_type in m.plant_types:
                try:
                    p = pyo.value(m.p_hourly[year, d, h, plant_type])
                    if p is None:
                        p = 0.0
                except:
                    p = 0.0
                total_gen += p * weight

    # Calculate percentage
    if total_gen > 0:
        nuclear_pct = (nuclear_gen / total_gen) * 100
    else:
        nuclear_pct = 0.0

    constraint_satisfied = nuclear_pct >= 49.9  # Allow tiny numerical tolerance
    status = '✓' if constraint_satisfied else '✗ VIOLATED'

    if not constraint_satisfied:
        all_satisfied = False

    print(f"{year:<8} {nuclear_gen:>17.2f} {total_gen:>17.2f} {nuclear_pct:>11.1f}% {status:>12}")

print("="*80)

if all_satisfied:
    print("\n✓ SUCCESS: All years meet the 50% nuclear generation requirement!")
else:
    print("\n✗ FAILURE: Some years do NOT meet the 50% nuclear requirement!")
    print("   The constraint is not working correctly.")

print("\n" + "="*80)
