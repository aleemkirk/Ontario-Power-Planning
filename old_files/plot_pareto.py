"""
Quick plot of the Pareto frontier from CSV file.
"""

import pandas as pd
import matplotlib.pyplot as plt

# Read the Pareto frontier data
df = pd.read_csv('results/data/pareto_frontier.csv')

print("Pareto Frontier Solutions:")
print("=" * 70)
print(df.to_string(index=False))

# Create visualization
fig, ax = plt.subplots(figsize=(12, 8))

# Plot Pareto frontier
ax.plot(df['emissions_megatons'], df['cost_billions'],
        'o-', linewidth=3, markersize=10, color='#2E86AB', label='Pareto Frontier')

# Annotate endpoints
min_cost_idx = df['cost_billions'].idxmin()
min_emis_idx = df['emissions_megatons'].idxmin()

ax.annotate(f'Cost-Optimal\n${df.loc[min_cost_idx, "cost_billions"]:.1f}B\n{df.loc[min_cost_idx, "emissions_megatons"]:.0f} MT CO2',
            xy=(df.loc[min_cost_idx, 'emissions_megatons'],
                df.loc[min_cost_idx, 'cost_billions']),
            xytext=(50, -30), textcoords='offset points',
            fontsize=11, fontweight='bold', color='#A23B72',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='yellow', alpha=0.8),
            arrowprops=dict(arrowstyle='->', color='#A23B72', lw=2))

ax.annotate(f'Emissions-Optimal\n${df.loc[min_emis_idx, "cost_billions"]:.1f}B\n{df.loc[min_emis_idx, "emissions_megatons"]:.0f} MT CO2',
            xy=(df.loc[min_emis_idx, 'emissions_megatons'],
                df.loc[min_emis_idx, 'cost_billions']),
            xytext=(-100, 20), textcoords='offset points',
            fontsize=11, fontweight='bold', color='#18A558',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgreen', alpha=0.8),
            arrowprops=dict(arrowstyle='->', color='#18A558', lw=2))

ax.set_xlabel('Total Emissions (megatons CO2)', fontsize=14, fontweight='bold')
ax.set_ylabel('Total Cost ($ billions NPV)', fontsize=14, fontweight='bold')
ax.set_title('Pareto Frontier: Cost vs Emissions Trade-off\nOntario Power Planning (2025-2045)',
            fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=12, loc='upper right')

plt.tight_layout()
plt.savefig('results/figures/pareto_frontier.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved plot to results/figures/pareto_frontier.png")

# Calculate and display key insights
print(f"\n{'=' * 70}")
print("Key Insights:")
print(f"{'=' * 70}\n")

min_cost = df.loc[min_cost_idx]
min_emis = df.loc[min_emis_idx]

cost_increase = min_emis['cost_billions'] - min_cost['cost_billions']
emis_reduction = min_cost['emissions_megatons'] - min_emis['emissions_megatons']
marginal_cost = (cost_increase * 1e9) / (emis_reduction * 1e6)

print(f"1. Range of Solutions:")
print(f"   Cost: ${min_cost['cost_billions']:.1f}B - ${min_emis['cost_billions']:.1f}B")
print(f"   Emissions: {min_emis['emissions_megatons']:.1f} MT - {min_cost['emissions_megatons']:.1f} MT")

print(f"\n2. Trade-off:")
print(f"   To reduce emissions by {emis_reduction/min_cost['emissions_megatons']*100:.1f}%")
print(f"   Requires {cost_increase/min_cost['cost_billions']*100:.1f}% more investment (${cost_increase:.1f}B)")
print(f"   Marginal cost: ${marginal_cost:.0f}/ton CO2")

print(f"\n3. Capacity Requirements:")
print(f"   Cost-optimal needs: {min_cost['new_capacity_GW']:.1f} GW new capacity")
print(f"   Emissions-optimal needs: {min_emis['new_capacity_GW']:.1f} GW new capacity")
print(f"   Difference: {min_emis['new_capacity_GW'] - min_cost['new_capacity_GW']:.1f} GW more")

print(f"\n4. Recommendation:")
mid_idx = len(df) // 2
mid = df.iloc[mid_idx]
print(f"   Balanced solution (α={mid['alpha']:.2f}):")
print(f"   Cost: ${mid['cost_billions']:.1f}B")
print(f"   Emissions: {mid['emissions_megatons']:.1f} MT ({(1-mid['emissions_megatons']/min_cost['emissions_megatons'])*100:.0f}% reduction)")
print(f"   New capacity: {mid['new_capacity_GW']:.1f} GW")

plt.show()
