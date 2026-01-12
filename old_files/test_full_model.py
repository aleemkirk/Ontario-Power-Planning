"""
Test script for the full 20-year model.

Tests the complete model (2025-2045) with cost minimization.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.optimization.model import PowerSystemOptimization

print("=" * 70)
print("Ontario Power Plant Optimization - Full 20-Year Model")
print("=" * 70)

# Create model with full 20-year horizon
model = PowerSystemOptimization(
    start_year=2025,
    end_year=2045,  # Full 20 years
    data_path='data/processed/'
)

# Run optimization
print("\nRunning 20-year optimization...")
print("This may take a few minutes...")
results = model.optimize(objective='cost', solver='highs', time_limit=300)

if results is not None:
    print("\n" + "=" * 70)
    print("SUCCESS! Full 20-year model working correctly")
    print("=" * 70)

    # Display detailed results
    print("\n[New Capacity Builds by Year and Type (showing non-zero only)]")
    builds = results['new_builds']
    nonzero_builds = builds[builds['new_capacity_MW'] > 1.0]  # Show builds > 1 MW
    if len(nonzero_builds) > 0:
        print(nonzero_builds.to_string(index=False))
    else:
        print("  No new builds needed")

    print("\n[Total New Capacity by Plant Type]")
    new_by_type = results['new_builds'].groupby('plant_type')['new_capacity_MW'].sum()
    for plant, cap in new_by_type.items():
        if cap > 1.0:
            print(f"  {plant}: {cap/1000:.2f} GW")

    print("\n[Capacity Growth]")
    cap_2025 = results['capacity'][results['capacity']['year'] == 2025]['total_capacity_MW'].sum()
    cap_2045 = results['capacity'][results['capacity']['year'] == 2045]['total_capacity_MW'].sum()
    print(f"  2025: {cap_2025/1000:.2f} GW")
    print(f"  2045: {cap_2045/1000:.2f} GW")
    print(f"  Growth: {((cap_2045/cap_2025 - 1) * 100):.1f}%")

    print("\n[Generation Mix 2045]")
    gen_2045 = results['generation'][results['generation']['year'] == 2045]
    for _, row in gen_2045.iterrows():
        pct = row['generation_MWh'] / gen_2045['generation_MWh'].sum() * 100
        print(f"  {row['plant_type']}: {row['generation_MWh']/1e6:.2f} TWh ({pct:.1f}%)")

    # Save results
    results['new_builds'].to_csv('results/data/new_builds.csv', index=False)
    results['capacity'].to_csv('results/data/capacity.csv', index=False)
    results['generation'].to_csv('results/data/generation.csv', index=False)
    print("\n✓ Results saved to results/data/")

else:
    print("\n✗ Optimization failed!")
    sys.exit(1)
