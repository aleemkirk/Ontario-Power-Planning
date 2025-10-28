"""
Quick test script for the prototype model.

Tests a 5-year model (2025-2030) with cost minimization.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.optimization.model import PowerSystemOptimization

print("=" * 70)
print("Ontario Power Plant Optimization - Prototype Test")
print("=" * 70)

# Create model with 5-year horizon (prototype)
model = PowerSystemOptimization(
    start_year=2025,
    end_year=2029,  # 5 years for testing
    data_path='data/processed/'
)

# Run optimization
print("\nRunning 5-year prototype optimization...")
results = model.optimize(objective='cost', solver='highs', time_limit=60)

if results is not None:
    print("\n" + "=" * 70)
    print("SUCCESS! Prototype model working correctly")
    print("=" * 70)

    # Display detailed results
    print("\n[New Capacity Builds by Year and Type]")
    builds_summary = results['new_builds'].pivot(
        index='year',
        columns='plant_type',
        values='new_capacity_MW'
    )
    print(builds_summary.to_string())

    print("\n[Total Capacity by Year]")
    capacity_by_year = results['capacity'].groupby('year')['total_capacity_MW'].sum()
    for year, cap in capacity_by_year.items():
        print(f"  {year}: {cap/1000:.2f} GW")

    print("\n[Generation by Plant Type (Total)]")
    gen_by_type = results['generation'].groupby('plant_type')['generation_MWh'].sum()
    for plant, gen in gen_by_type.items():
        print(f"  {plant}: {gen/1e6:.2f} TWh")

else:
    print("\n✗ Optimization failed!")
    sys.exit(1)
