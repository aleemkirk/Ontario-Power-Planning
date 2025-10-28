"""
Generate and visualize the Pareto frontier.

This creates 10-15 optimal solutions showing cost-emissions trade-offs.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.optimization.model import PowerSystemOptimization
from src.analysis.pareto import generate_pareto_frontier
import matplotlib.pyplot as plt
import pandas as pd

print("=" * 70)
print("Pareto Frontier Generation")
print("=" * 70)

# Generate Pareto frontier
pareto_df = generate_pareto_frontier(
    model_class=PowerSystemOptimization,
    n_points=10,  # Start with 10 points
    method='weighted_sum',
    solver='highs',
    # Model arguments
    start_year=2025,
    end_year=2045,
    data_path='data/processed/',
    use_lead_times=True,
    use_retirements=True
)

# Save results
pareto_df.to_csv('results/data/pareto_frontier.csv', index=False)
print(f"\n✓ Saved to results/data/pareto_frontier.csv")

# Display results
print(f"\n{'=' * 70}")
print("Pareto Frontier Solutions")
print(f"{'=' * 70}\n")
print(pareto_df.to_string(index=False))

# Calculate marginal costs
print(f"\n{'=' * 70}")
print("Marginal Abatement Cost Analysis")
print(f"{'=' * 70}\n")

pareto_sorted = pareto_df.sort_values('emissions_megatons')
for i in range(len(pareto_sorted) - 1):
    current = pareto_sorted.iloc[i]
    next_sol = pareto_sorted.iloc[i + 1]

    cost_increase = next_sol['cost_billions'] - current['cost_billions']
    emis_reduction = current['emissions_megatons'] - next_sol['emissions_megatons']

    if emis_reduction > 0:
        marginal_cost = (cost_increase * 1e9) / (emis_reduction * 1e6)
        print(f"  {current['emissions_megatons']:.1f} MT → {next_sol['emissions_megatons']:.1f} MT: ${marginal_cost:.0f}/ton CO2")

# Create visualization
print(f"\n{'=' * 70}")
print("Creating Visualization")
print(f"{'=' * 70}\n")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Pareto frontier
ax1 = axes[0]
ax1.plot(pareto_df['emissions_megatons'], pareto_df['cost_billions'],
         'o-', linewidth=2, markersize=8, color='#2E86AB')
ax1.set_xlabel('Total Emissions (megatons CO2)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Total Cost ($ billions NPV)', fontsize=12, fontweight='bold')
ax1.set_title('Pareto Frontier: Cost vs Emissions\nOntario Power Planning (2025-2045)',
              fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Annotate endpoints
min_cost_idx = pareto_df['cost_billions'].idxmin()
min_emis_idx = pareto_df['emissions_megatons'].idxmin()

ax1.annotate('Cost-Optimal\n(Gas-heavy)',
             xy=(pareto_df.loc[min_cost_idx, 'emissions_megatons'],
                 pareto_df.loc[min_cost_idx, 'cost_billions']),
             xytext=(10, 10), textcoords='offset points',
             fontsize=10, fontweight='bold', color='#A23B72',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
             arrowprops=dict(arrowstyle='->', color='#A23B72', lw=2))

ax1.annotate('Emissions-Optimal\n(Renewables)',
             xy=(pareto_df.loc[min_emis_idx, 'emissions_megatons'],
                 pareto_df.loc[min_emis_idx, 'cost_billions']),
             xytext=(10, -30), textcoords='offset points',
             fontsize=10, fontweight='bold', color='#18A558',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7),
             arrowprops=dict(arrowstyle='->', color='#18A558', lw=2))

# Plot 2: Capacity vs Emissions
ax2 = axes[1]
ax2.plot(pareto_df['emissions_megatons'], pareto_df['new_capacity_GW'],
         's-', linewidth=2, markersize=8, color='#F18F01')
ax2.set_xlabel('Total Emissions (megatons CO2)', fontsize=12, fontweight='bold')
ax2.set_ylabel('New Capacity Built (GW)', fontsize=12, fontweight='bold')
ax2.set_title('Capacity Needed vs Emissions Target', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/pareto_frontier.png', dpi=300, bbox_inches='tight')
print("✓ Saved plot to results/figures/pareto_frontier.png")

plt.show()

print(f"\n{'=' * 70}")
print("Key Policy Insights")
print(f"{'=' * 70}\n")

min_cost_sol = pareto_df.loc[pareto_df['cost_billions'].idxmin()]
min_emis_sol = pareto_df.loc[pareto_df['emissions_megatons'].idxmin()]

print(f"1. Cost-Optimal Solution:")
print(f"   Cost: ${min_cost_sol['cost_billions']:.1f}B")
print(f"   Emissions: {min_cost_sol['emissions_megatons']:.1f} MT CO2")
print(f"   New capacity: {min_cost_sol['new_capacity_GW']:.1f} GW")

print(f"\n2. Emissions-Optimal Solution:")
print(f"   Cost: ${min_emis_sol['cost_billions']:.1f}B (+${min_emis_sol['cost_billions'] - min_cost_sol['cost_billions']:.1f}B)")
print(f"   Emissions: {min_emis_sol['emissions_megatons']:.1f} MT CO2 (-{min_cost_sol['emissions_megatons'] - min_emis_sol['emissions_megatons']:.1f} MT)")
print(f"   New capacity: {min_emis_sol['new_capacity_GW']:.1f} GW")

print(f"\n3. Trade-off:")
emissions_reduction_pct = (1 - min_emis_sol['emissions_megatons'] / min_cost_sol['emissions_megatons']) * 100
cost_increase_pct = (min_emis_sol['cost_billions'] / min_cost_sol['cost_billions'] - 1) * 100
print(f"   Reducing emissions by {emissions_reduction_pct:.0f}% costs {cost_increase_pct:.0f}% more")

print(f"\n4. Recommended Scenario:")
# Find a balanced point (around 50% emissions reduction)
mid_point = pareto_df.iloc[len(pareto_df) // 2]
print(f"   Balanced approach at α=0.5:")
print(f"   Cost: ${mid_point['cost_billions']:.1f}B")
print(f"   Emissions: {mid_point['emissions_megatons']:.1f} MT CO2")
print(f"   (~{(1 - mid_point['emissions_megatons']/min_cost_sol['emissions_megatons'])*100:.0f}% emissions reduction)")
