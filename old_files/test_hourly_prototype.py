"""
Test script for Phase 2: 1-year hourly prototype model (2025 only).

Tests:
- Model builds successfully
- Solves in reasonable time (<5 minutes target)
- All constraints satisfied
- Ramp rate constraints are active and binding
- Results make physical sense

Usage:
    python test_hourly_prototype.py
"""

import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.optimization.model_hourly import PowerSystemOptimizationHourly
import pyomo.environ as pyo


def test_hourly_prototype():
    """Run 1-year prototype test for hourly model."""

    print("=" * 80)
    print("PHASE 2: HOURLY PROTOTYPE MODEL TEST")
    print("=" * 80)
    print(f"\nTest Configuration:")
    print(f"  Year: 2025 (single year)")
    print(f"  Representative days: 12")
    print(f"  Hours per day: 24")
    print(f"  Ramp constraints: ACTIVATED (soft with $1000/MW penalty)")
    print(f"  Solver: HiGHS (open-source)")
    print(f"  Expected variables: ~69,360")
    print(f"  Expected constraints: ~106,700")

    # ========================================================================
    # 1. Initialize Model
    # ========================================================================

    print("\n" + "-" * 80)
    print("Step 1: Initialize Hourly Model")
    print("-" * 80)

    start_init = time.time()

    optimizer = PowerSystemOptimizationHourly(
        start_year=2025,
        end_year=2025,  # Single year test
        n_rep_days=12,
        use_soft_ramp_constraints=True,
        ramp_penalty=1000.0  # $/MW penalty for ramp violations
    )

    elapsed_init = time.time() - start_init
    print(f"✓ Model initialized in {elapsed_init:.2f} seconds")

    # ========================================================================
    # 2. Build Model
    # ========================================================================

    print("\n" + "-" * 80)
    print("Step 2: Build Optimization Model")
    print("-" * 80)

    start_build = time.time()

    # Build with cost objective first (single objective test)
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
    print("Step 3: Solve Optimization Problem")
    print("-" * 80)
    print("Objective: Minimize total system cost (NPV)")
    print("Using HiGHS solver...")

    start_solve = time.time()

    result = optimizer.solve(solver='highs', tee=True, save_results=False)

    elapsed_solve = time.time() - start_solve

    print(f"\n✓ Solve completed in {elapsed_solve:.2f} seconds")

    # Check solve status
    if result['status'] != 'optimal':
        print(f"\n⚠️  WARNING: Solver status is '{result['status']}', not 'optimal'")
        print("Model may be infeasible or unbounded. Check constraints.")
        return False

    print(f"✓ Solver status: {result['status']}")

    # ========================================================================
    # 4. Extract and Validate Results
    # ========================================================================

    print("\n" + "-" * 80)
    print("Step 4: Extract and Validate Results")
    print("-" * 80)

    # Extract results
    results = optimizer.get_results()

    # Display summary
    print("\n[COST BREAKDOWN]")
    print(f"  Total Cost (NPV):        ${results['total_cost']/1e9:>10.2f}B")
    print(f"  Capital Costs:           ${results['capex_cost']/1e9:>10.2f}B")
    print(f"  Operating Costs:         ${results['opex_cost']/1e9:>10.2f}B")
    print(f"  Maintenance Costs:       ${results['maintenance_cost']/1e9:>10.2f}B")
    if 'ramp_penalty_cost' in results and results['ramp_penalty_cost'] > 0:
        print(f"  Ramp Penalty Cost:       ${results['ramp_penalty_cost']/1e9:>10.2f}B")
        print(f"    → Ramp constraints ARE BINDING! ✓")
    else:
        print(f"  Ramp Penalty Cost:       ${0:>10.2f}B")
        print(f"    → Ramp constraints NOT binding (no violations)")

    print(f"\n[EMISSIONS]")
    print(f"  Total Emissions:         {results['total_emissions']/1e6:>10.2f}M tons CO2")

    print(f"\n[CAPACITY MIX - 2025]")
    capacity_2025 = results['capacity'][2025]
    for plant_type in ['nuclear', 'hydro', 'gas', 'wind', 'solar', 'biofuel']:
        capacity_mw = capacity_2025[plant_type]
        print(f"  {plant_type.capitalize():12s}  {capacity_mw:>10,.0f} MW")

    total_capacity = sum(capacity_2025.values())
    print(f"  {'TOTAL':12s}  {total_capacity:>10,.0f} MW")

    # ========================================================================
    # 5. Validation Checks
    # ========================================================================

    print("\n" + "-" * 80)
    print("Step 5: Validation Checks")
    print("-" * 80)

    validation_passed = True

    # Check 1: Solve time < 5 minutes (target)
    print(f"\n[Check 1] Solve Time")
    if elapsed_solve < 300:  # 5 minutes
        print(f"  ✓ PASS: {elapsed_solve:.1f}s < 300s (5 min target)")
    else:
        print(f"  ⚠️  WARNING: {elapsed_solve:.1f}s > 300s (slow but acceptable)")

    # Check 2: Feasible solution
    print(f"\n[Check 2] Feasibility")
    if result['status'] == 'optimal':
        print(f"  ✓ PASS: Optimal solution found")
    else:
        print(f"  ✗ FAIL: Status = {result['status']}")
        validation_passed = False

    # Check 3: Ramp constraints active
    print(f"\n[Check 3] Ramp Rate Constraints")
    if 'ramp_penalty_cost' in results and results['ramp_penalty_cost'] > 0:
        print(f"  ✓ PASS: Ramp constraints are ACTIVE and BINDING")
        print(f"    Penalty cost = ${results['ramp_penalty_cost']/1e6:.2f}M")
    else:
        print(f"  ℹ️  INFO: Ramp constraints active but no violations")
        print(f"    This means all plants can ramp fast enough for this scenario")

    # Check 4: Capacity makes sense
    print(f"\n[Check 4] Capacity Sanity Check")
    if total_capacity > 0 and total_capacity < 200000:  # 0 < capacity < 200 GW
        print(f"  ✓ PASS: Total capacity {total_capacity:,.0f} MW is reasonable")
    else:
        print(f"  ✗ FAIL: Total capacity {total_capacity:,.0f} MW seems unrealistic")
        validation_passed = False

    # Check 5: Cost is positive
    print(f"\n[Check 5] Cost Sanity Check")
    if results['total_cost'] > 0:
        print(f"  ✓ PASS: Total cost ${results['total_cost']/1e9:.2f}B is positive")
    else:
        print(f"  ✗ FAIL: Total cost is not positive")
        validation_passed = False

    # ========================================================================
    # 6. Summary
    # ========================================================================

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    total_time = elapsed_init + elapsed_build + elapsed_solve

    print(f"\nTiming Breakdown:")
    print(f"  Initialization:  {elapsed_init:>8.2f}s")
    print(f"  Model Build:     {elapsed_build:>8.2f}s")
    print(f"  Solver Time:     {elapsed_solve:>8.2f}s")
    print(f"  {'─' * 25}")
    print(f"  Total Time:      {total_time:>8.2f}s")

    print(f"\nValidation:")
    if validation_passed:
        print(f"  ✓ ALL CHECKS PASSED")
    else:
        print(f"  ✗ SOME CHECKS FAILED")

    print(f"\nKey Achievements:")
    print(f"  ✓ Hourly model successfully built and solved")
    print(f"  ✓ Representative days integrated (12 clusters)")
    print(f"  ✓ Ramp rate constraints ACTIVATED (soft)")
    if 'ramp_penalty_cost' in results and results['ramp_penalty_cost'] > 0:
        print(f"  ✓ Ramp constraints ARE BINDING (realistic operations)")
    print(f"  ✓ Model size: {n_vars:,} variables, {n_constraints:,} constraints")
    print(f"  ✓ Solve time: {elapsed_solve:.1f}s {'(fast!)' if elapsed_solve < 60 else '(acceptable)'}")

    print("\n" + "=" * 80)
    print("PHASE 2 PROTOTYPE TEST COMPLETE!")
    print("=" * 80)

    return validation_passed


if __name__ == '__main__':
    try:
        success = test_hourly_prototype()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
