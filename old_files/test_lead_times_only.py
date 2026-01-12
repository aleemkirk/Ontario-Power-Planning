"""
Test hourly model with lead times only (no retirements).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.optimization.model_hourly import PowerSystemOptimizationHourly
import pyomo.environ as pyo

print("Testing hourly model with lead times but NO retirements...")
print("="*80)

optimizer = PowerSystemOptimizationHourly(
    start_year=2025,
    end_year=2045,
    use_lead_times=True,   # Enable lead times
    use_retirements=False, # Disable retirements
    n_rep_days=12
)

optimizer.build_model(objective='cost')

opt = pyo.SolverFactory('appsi_highs')
opt.config.load_solution = False

print("Solving...")
try:
    result = opt.solve(optimizer.model)
    print("✓ FEASIBLE with lead times (no retirements)")
    obj_val = pyo.value(optimizer.model.obj)
    print(f"Objective value: ${obj_val/1e9:.2f}B")
except RuntimeError as e:
    print("✗ INFEASIBLE even without retirements")
    print(f"Error: {e}")

print("\n" + "="*80)
print("Testing hourly model with retirements but NO lead times...")
print("="*80)

optimizer2 = PowerSystemOptimizationHourly(
    start_year=2025,
    end_year=2045,
    use_lead_times=False,  # Disable lead times
    use_retirements=True,  # Enable retirements
    n_rep_days=12
)

optimizer2.build_model(objective='cost')

print("Solving...")
try:
    result2 = opt.solve(optimizer2.model)
    print("✓ FEASIBLE with retirements (no lead times)")
    obj_val = pyo.value(optimizer2.model.obj)
    print(f"Objective value: ${obj_val/1e9:.2f}B")
except RuntimeError as e:
    print("✗ INFEASIBLE with retirements")
    print(f"Error: {e}")
