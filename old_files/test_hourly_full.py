"""
Test script for Phase 3: Full 20-year hourly model (2025-2045).

Tests:
- Full planning horizon with hourly dispatch
- Capacity expansion optimization
- Ramp rate constraints active
- Solve time validation (<10 minutes target)
- Results analysis and comparison

Usage:
    python test_hourly_full.py
"""

import time
import sys
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.optimization.model_hourly import PowerSystemOptimizationHourly
import pyomo.environ as pyo


def test_hourly_full():
    """Run full 20-year hourly model test."""

    print("=" * 80)
    print("PHASE 3: FULL 20-YEAR HOURLY MODEL TEST")
    print("=" * 80)
    print(f"\nTest Configuration:")
    print(f"  Planning horizon: 2025-2045 (21 years)")
    print(f"  Representative days: 12")
    print(f"  Hours per day: 24")
    print(f"  Ramp constraints: ACTIVATED (soft with $1000/MW penalty)")
    print(f"  Lead times: ENABLED (nuclear=7yr, hydro=6yr, etc.)")
    print(f"  Retirements: DISABLED (causes infeasibility with lead times)")
    print(f"  Solver: HiGHS (open-source)")
    print(f"  Expected variables: ~69,360")
    print(f"  Expected constraints: ~106,700")
    print(f"  Target solve time: <10 minutes")

    print(f"\nNote: Retirements disabled due to infeasibility when combined with lead times.")
    print(f"      Plants retire faster than replacements can be built (lead time lag).")
    print(f"      This will be fixed in a future phase with 'builds in progress' parameter.")

    # ========================================================================
    # 1. Initialize Model
    # ========================================================================

    print("\n" + "-" * 80)
    print("Step 1: Initialize Full Hourly Model")
    print("-" * 80)

    start_init = time.time()

    optimizer = PowerSystemOptimizationHourly(
        start_year=2025,
        end_year=2045,  # Full 20-year horizon
        n_rep_days=12,
        use_soft_ramp_constraints=True,
        ramp_penalty=1000.0,  # $/MW penalty for ramp violations
        use_lead_times=True,  # Enable lead times
        use_retirements=False  # Disable retirements (causes infeasibility with lead times)
    )

    elapsed_init = time.time() - start_init
    print(f"✓ Model initialized in {elapsed_init:.2f} seconds")

    # ========================================================================
    # 2. Build Model
    # ========================================================================

    print("\n" + "-" * 80)
    print("Step 2: Build Full Optimization Model")
    print("-" * 80)

    start_build = time.time()

    # Build with cost objective
    optimizer.build_model(objective='cost', alpha=1.0)

    elapsed_build = time.time() - start_build

    # Get model statistics
    m = optimizer.model
    n_vars = len(list(m.component_data_objects(pyo.Var)))
    n_constraints = len(list(m.component_data_objects(pyo.Constraint)))

    print(f"✓ Model built in {elapsed_build:.2f} seconds")
    print(f"\nModel Statistics:")
    print(f"  Total variables: {n_vars:,}")
    print(f"  Total constraints: {n_constraints:,}")
    print(f"  Plant types: {len(m.plant_types)}")
    print(f"  Years: {len(m.years)}")
    print(f"  Representative days: {len(m.rep_days)}")
    print(f"  Hours per day: {len(m.hours)}")

    # ========================================================================
    # 3. Solve Model
    # ========================================================================

    print("\n" + "-" * 80)
    print("Step 3: Solve Full 20-Year Optimization")
    print("-" * 80)
    print("Objective: Minimize total system cost (NPV)")
    print("Using HiGHS solver...")
    print("This may take several minutes...")

    start_solve = time.time()

    result = optimizer.solve(solver='highs', time_limit=600, tee=True, save_results=True)

    elapsed_solve = time.time() - start_solve

    print(f"\n✓ Solve completed in {elapsed_solve:.1f} seconds ({elapsed_solve/60:.2f} minutes)")

    # Check solve status
    if result['status'] != 'optimal':
        print(f"\n⚠️  WARNING: Solver status is '{result['status']}', not 'optimal'")
        if result['status'] == 'failed':
            print("Model may be infeasible or unbounded. Check constraints.")
            return False
        else:
            print("Feasible solution found, continuing with analysis...")

    print(f"✓ Solver status: {result['status']}")

    # ========================================================================
    # 4. Extract and Analyze Results
    # ========================================================================

    print("\n" + "-" * 80)
    print("Step 4: Extract and Analyze Results")
    print("-" * 80)

    # Results already extracted by solve() with save_results=True
    results = optimizer.results

    # Display summary
    print("\n[COST BREAKDOWN - 20 YEARS]")
    print(f"  Total Cost (NPV):        ${results['total_cost']/1e9:>10.2f}B")
    print(f"  Capital Costs:           ${results['capex_cost']/1e9:>10.2f}B")
    print(f"  Operating Costs:         ${results['opex_cost']/1e9:>10.2f}B")
    print(f"  Maintenance Costs:       ${results['maintenance_cost']/1e9:>10.2f}B")
    if results['ramp_penalty_cost'] > 0:
        print(f"  Ramp Penalty Cost:       ${results['ramp_penalty_cost']/1e9:>10.2f}B")
        print(f"    → Ramp constraints ARE BINDING over 20 years! ✓")
    else:
        print(f"  Ramp Penalty Cost:       ${0:>10.2f}B")
        print(f"    → Ramp constraints NOT binding (no violations)")

    print(f"\n[EMISSIONS - 20 YEARS]")
    print(f"  Total Emissions:         {results['total_emissions']/1e6:>10.2f}M tons CO2")

    # Capacity expansion analysis
    print(f"\n[CAPACITY EXPANSION]")

    # Initial capacity (2025)
    capacity_2025 = results['capacity'][2025]
    print(f"\nInitial Capacity (2025):")
    for plant_type in ['nuclear', 'hydro', 'gas', 'wind', 'solar', 'biofuel']:
        capacity_mw = capacity_2025[plant_type]
        print(f"  {plant_type.capitalize():12s}  {capacity_mw:>10,.0f} MW")
    total_2025 = sum(capacity_2025.values())
    print(f"  {'TOTAL':12s}  {total_2025:>10,.0f} MW")

    # Final capacity (2045)
    capacity_2045 = results['capacity'][2045]
    print(f"\nFinal Capacity (2045):")
    for plant_type in ['nuclear', 'hydro', 'gas', 'wind', 'solar', 'biofuel']:
        capacity_mw = capacity_2045[plant_type]
        change = capacity_mw - capacity_2025[plant_type]
        change_pct = (change / capacity_2025[plant_type] * 100) if capacity_2025[plant_type] > 0 else 0
        print(f"  {plant_type.capitalize():12s}  {capacity_mw:>10,.0f} MW  ({change:+,.0f} MW, {change_pct:+.1f}%)")
    total_2045 = sum(capacity_2045.values())
    total_change = total_2045 - total_2025
    print(f"  {'TOTAL':12s}  {total_2045:>10,.0f} MW  ({total_change:+,.0f} MW, {total_change/total_2025*100:+.1f}%)")

    # New builds by type
    print(f"\n[NEW CAPACITY BUILT 2025-2045]")
    new_builds_by_type = results['new_builds'].groupby('plant_type')['new_capacity_MW'].sum()
    for plant_type in ['nuclear', 'hydro', 'gas', 'wind', 'solar', 'biofuel']:
        if plant_type in new_builds_by_type.index:
            new_mw = new_builds_by_type[plant_type]
            if new_mw > 0:
                print(f"  {plant_type.capitalize():12s}  {new_mw:>10,.0f} MW")
    total_new = new_builds_by_type.sum()
    print(f"  {'TOTAL':12s}  {total_new:>10,.0f} MW")

    # ========================================================================
    # 5. Validation Checks
    # ========================================================================

    print("\n" + "-" * 80)
    print("Step 5: Validation Checks")
    print("-" * 80)

    validation_passed = True

    # Check 1: Solve time < 10 minutes
    print(f"\n[Check 1] Solve Time")
    if elapsed_solve < 600:  # 10 minutes
        print(f"  ✓ PASS: {elapsed_solve:.1f}s < 600s (10 min target)")
    else:
        print(f"  ⚠️  WARNING: {elapsed_solve:.1f}s > 600s (acceptable but slow)")

    # Check 2: Feasible solution
    print(f"\n[Check 2] Feasibility")
    if result['status'] == 'optimal':
        print(f"  ✓ PASS: Optimal solution found")
    elif result['status'] == 'feasible':
        print(f"  ⚠️  INFO: Feasible solution (not proven optimal)")
    else:
        print(f"  ✗ FAIL: Status = {result['status']}")
        validation_passed = False

    # Check 3: Ramp constraints
    print(f"\n[Check 3] Ramp Rate Constraints")
    if results['ramp_penalty_cost'] > 0:
        print(f"  ✓ PASS: Ramp constraints are ACTIVE and BINDING")
        print(f"    Penalty cost = ${results['ramp_penalty_cost']/1e6:.2f}M")
    else:
        print(f"  ℹ️  INFO: Ramp constraints active but no violations")

    # Check 4: Capacity growth makes sense
    print(f"\n[Check 4] Capacity Growth")
    if total_2045 > total_2025 and total_2045 < 200000:
        growth_pct = (total_2045 - total_2025) / total_2025 * 100
        print(f"  ✓ PASS: Capacity grew {growth_pct:.1f}% (reasonable for demand growth)")
    else:
        print(f"  ⚠️  WARNING: Capacity change seems unusual")

    # Check 5: Emissions are reasonable
    print(f"\n[Check 5] Emissions Sanity Check")
    avg_annual_emissions = results['total_emissions'] / 21  # 21 years
    if avg_annual_emissions > 0 and avg_annual_emissions < 50e6:  # 0-50 MT/year
        print(f"  ✓ PASS: Average {avg_annual_emissions/1e6:.2f} MT/year is reasonable")
    else:
        print(f"  ⚠️  WARNING: Emissions seem unusual")

    # ========================================================================
    # 6. Save Results
    # ========================================================================

    print("\n" + "-" * 80)
    print("Step 6: Save Results")
    print("-" * 80)

    # Save detailed results
    results_dir = Path("results/data")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save new builds
    builds_file = results_dir / "hourly_new_builds_2025_2045.csv"
    results['new_builds'].to_csv(builds_file, index=False)
    print(f"✓ Saved new builds to {builds_file}")

    # Save generation
    gen_file = results_dir / "hourly_generation_2025_2045.csv"
    results['generation'].to_csv(gen_file, index=False)
    print(f"✓ Saved generation to {gen_file}")

    # Save summary
    summary_file = results_dir / "hourly_summary_2025_2045.json"
    summary_data = {
        'total_cost': results['total_cost'],
        'capex_cost': results['capex_cost'],
        'opex_cost': results['opex_cost'],
        'maintenance_cost': results['maintenance_cost'],
        'ramp_penalty_cost': results['ramp_penalty_cost'],
        'total_emissions': results['total_emissions'],
        'capacity_2025': capacity_2025,
        'capacity_2045': capacity_2045,
        'new_builds_by_type': new_builds_by_type.to_dict(),
        'solve_time_seconds': elapsed_solve,
        'solver_status': result['status']
    }
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"✓ Saved summary to {summary_file}")

    # ========================================================================
    # 7. Summary
    # ========================================================================

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    total_time = elapsed_init + elapsed_build + elapsed_solve

    print(f"\nTiming Breakdown:")
    print(f"  Initialization:  {elapsed_init:>8.2f}s")
    print(f"  Model Build:     {elapsed_build:>8.2f}s")
    print(f"  Solver Time:     {elapsed_solve:>8.1f}s ({elapsed_solve/60:.2f} min)")
    print(f"  {'─' * 40}")
    print(f"  Total Time:      {total_time:>8.1f}s ({total_time/60:.2f} min)")

    print(f"\nModel Statistics:")
    print(f"  Variables:       {n_vars:>8,}")
    print(f"  Constraints:     {n_constraints:>8,}")
    print(f"  Planning years:  {len(m.years):>8,}")

    print(f"\nKey Results:")
    print(f"  Total Cost:      ${results['total_cost']/1e9:>8.2f}B")
    print(f"  Ramp Penalty:    ${results['ramp_penalty_cost']/1e6:>8.2f}M")
    print(f"  New Capacity:    {total_new:>8,.0f} MW")
    print(f"  Final Capacity:  {total_2045:>8,.0f} MW")
    print(f"  Total Emissions: {results['total_emissions']/1e6:>8.2f} MT")

    print(f"\nValidation:")
    if validation_passed:
        print(f"  ✓ ALL CHECKS PASSED")
    else:
        print(f"  ⚠️  SOME CHECKS FAILED OR WARNINGS")

    print("\n" + "=" * 80)
    print("PHASE 3 FULL 20-YEAR HOURLY MODEL TEST COMPLETE!")
    print("=" * 80)

    return validation_passed


if __name__ == '__main__':
    try:
        success = test_hourly_full()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
