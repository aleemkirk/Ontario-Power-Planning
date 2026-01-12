"""
Debug infeasibility in the full 20-year hourly model.

Strategy:
1. Try without some constraints to isolate the problem
2. Check capacity dynamics over time
3. Examine reserve margin requirements
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.optimization.model_hourly import PowerSystemOptimizationHourly
import pyomo.environ as pyo


def debug_infeasibility():
    """Debug why the 20-year model is infeasible."""

    print("=" * 80)
    print("DEBUGGING INFEASIBILITY")
    print("=" * 80)

    # Try building the model step by step
    print("\n[Test 1] Build model without lead times")
    print("-" * 80)

    optimizer1 = PowerSystemOptimizationHourly(
        start_year=2025,
        end_year=2045,
        use_lead_times=False,  # Disable lead times
        use_retirements=False,  # Disable retirements
        n_rep_days=12
    )

    optimizer1.build_model(objective='cost')

    # Try to solve
    opt = pyo.SolverFactory('appsi_highs')
    opt.config.load_solution = False  # Don't load solution if infeasible

    print("\nSolving without lead times or retirements...")
    try:
        result1 = opt.solve(optimizer1.model)
        print(f"✓ FEASIBLE without lead times/retirements")
        try:
            obj_val = pyo.value(optimizer1.model.obj)
            print(f"Objective value: ${obj_val/1e9:.2f}B")
        except:
            pass
    except RuntimeError as e:
        print(f"✗ Still INFEASIBLE - problem is more fundamental")
        print(f"Error: {e}")

    # Try with smaller horizon
    print("\n[Test 2] Build model with 5-year horizon")
    print("-" * 80)

    optimizer2 = PowerSystemOptimizationHourly(
        start_year=2025,
        end_year=2030,  # Just 5 years
        use_lead_times=True,
        use_retirements=True,
        n_rep_days=12
    )

    optimizer2.build_model(objective='cost')

    print("\nSolving 5-year model...")
    try:
        result2 = opt.solve(optimizer2.model)
        print("✓ FEASIBLE with 5 years")
        try:
            obj_val = pyo.value(optimizer2.model.obj)
            print(f"Objective value: ${obj_val/1e9:.2f}B")
        except:
            pass
        print("Issue is likely with long-term capacity dynamics or demand growth")
    except RuntimeError as e:
        print(f"✗ Still INFEASIBLE even with 5 years")
        print(f"Error: {e}")

    # Check demand growth vs capacity
    print("\n[Test 3] Examine Demand Growth")
    print("-" * 80)

    import json

    # Load demand forecast
    with open('data/processed/demand_forecast.csv') as f:
        import pandas as pd
        demand_df = pd.read_csv('data/processed/demand_forecast.csv')

    print("\nDemand Forecast:")
    print(demand_df)

    # Load initial capacity
    with open('data/processed/initial_capacity.json') as f:
        initial_cap = json.load(f)

    print("\nInitial Capacity (2025):")
    for plant, cap in initial_cap.items():
        print(f"  {plant}: {cap} MW")

    total_cap = sum(initial_cap.values())
    print(f"  TOTAL: {total_cap} MW")

    # Check if capacity can meet demand
    demand_2025 = demand_df[demand_df['year'] == 2025]['annual_demand_gwh'].values[0]
    demand_2045 = demand_df[demand_df['year'] == 2045]['annual_demand_gwh'].values[0]

    print(f"\nDemand 2025: {demand_2025:,.0f} GWh = {demand_2025/8760:.0f} MW average")
    print(f"Demand 2045: {demand_2045:,.0f} GWh = {demand_2045/8760:.0f} MW average")
    print(f"Demand growth: {(demand_2045/demand_2025 - 1)*100:.1f}%")

    # Capacity factors
    cf = {
        'nuclear': 0.90,
        'wind': 0.35,
        'solar': 0.15,
        'gas': 0.55,
        'hydro': 0.50,
        'biofuel': 0.80
    }

    # Effective generation capacity
    effective_cap = sum(initial_cap[p] * cf[p] for p in initial_cap.keys())
    print(f"\nEffective generation capacity (2025): {effective_cap:.0f} MW average")
    print(f"Average demand (2025): {demand_2025/8760:.0f} MW")
    print(f"Reserve margin: {(effective_cap / (demand_2025/8760) - 1)*100:.1f}%")

    # Check peak demand
    with open('data/processed/hourly_demand_2025_2045.json') as f:
        hourly_data = json.load(f)

    # Find peak demand in 2045
    rep_days_2045 = hourly_data['rep_days_by_year']['2045']
    peak_2045 = max(
        max(day_data['hourly_demand_mw'])
        for day_data in rep_days_2045.values()
    )

    print(f"\nPeak demand (2045): {peak_2045:,.0f} MW")
    print(f"Total capacity: {total_cap:,.0f} MW")
    print(f"Reserve margin: {(total_cap / peak_2045 - 1)*100:.1f}%")

    # Check if reserve margin constraint is too tight
    required_cap = peak_2045 * 1.15  # 15% reserve margin
    print(f"\nRequired capacity (2045, 15% reserve): {required_cap:,.0f} MW")
    print(f"Current capacity: {total_cap:,.0f} MW")
    print(f"Shortfall: {required_cap - total_cap:,.0f} MW")

    if required_cap > total_cap:
        print("\n⚠️  ISSUE FOUND: Current capacity cannot meet 2045 demand + reserve margin!")
        print("Need to build new capacity, but model may be over-constrained.")


if __name__ == '__main__':
    debug_infeasibility()
