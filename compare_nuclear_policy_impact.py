"""
Compare the impact of 50% nuclear policy on capacity expansion.

Creates comparison tables and visualizations showing how the nuclear policy
changes optimal capacity expansion decisions.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load both comparison tables
baseline = pd.read_csv('results/data/capacity_builds_comparison.csv', index_col=0)
nuclear_policy = pd.read_csv('results/data/capacity_builds_comparison_nuclear_policy.csv', index_col=0)

print("="*100)
print("IMPACT OF 50% NUCLEAR GENERATION POLICY ON CAPACITY EXPANSION")
print("="*100)

print("\n" + "="*100)
print("BASELINE (No Nuclear Policy)")
print("="*100)
print(baseline.to_string(float_format=lambda x: f'{x:>8.2f}'))

print("\n" + "="*100)
print("WITH 50% NUCLEAR POLICY")
print("="*100)
print(nuclear_policy.to_string(float_format=lambda x: f'{x:>8.2f}'))

# Calculate differences
print("\n" + "="*100)
print("DIFFERENCE (Nuclear Policy - Baseline) [GW]")
print("="*100)
difference = nuclear_policy - baseline
print(difference.to_string(float_format=lambda x: f'{x:>+8.2f}'))

# Key insights
print("\n" + "="*100)
print("KEY INSIGHTS: IMPACT OF 50% NUCLEAR POLICY")
print("="*100)

for col in baseline.columns:
    alpha = col.replace('α=', '')
    print(f"\n{col} ({['Emissions-Optimal', 'Balanced', 'Cost-Optimal'][int(float(alpha)*2)]})")
    print("-" * 80)

    # Nuclear change
    nuclear_change = difference.loc['nuclear', col] if 'nuclear' in difference.index else 0
    wind_change = difference.loc['wind', col] if 'wind' in difference.index else 0
    gas_change = difference.loc['gas', col] if 'gas' in difference.index else 0
    total_change = difference.loc['TOTAL', col]

    print(f"  Nuclear builds: {baseline.loc['nuclear', col]:>6.2f} GW → {nuclear_policy.loc['nuclear', col]:>6.2f} GW ({nuclear_change:>+6.2f} GW)")
    print(f"  Wind builds:    {baseline.loc['wind', col]:>6.2f} GW → {nuclear_policy.loc['wind', col]:>6.2f} GW ({wind_change:>+6.2f} GW)")
    print(f"  Gas builds:     {baseline.loc['gas', col]:>6.2f} GW → {nuclear_policy.loc['gas', col]:>6.2f} GW ({gas_change:>+6.2f} GW)")
    print(f"  Total builds:   {baseline.loc['TOTAL', col]:>6.2f} GW → {nuclear_policy.loc['TOTAL', col]:>6.2f} GW ({total_change:>+6.2f} GW)")

# Load cost and emissions data from saved results
print("\n" + "="*100)
print("COST AND EMISSIONS IMPACT")
print("="*100)

# Baseline costs and emissions (from previous runs)
baseline_data = {
    0.0: {'cost': 279.86, 'emissions': 49.02},
    0.5: {'cost': 147.89, 'emissions': 58.90},
    1.0: {'cost': 122.17, 'emissions': 315.79}
}

# Nuclear policy costs and emissions (from the run)
nuclear_policy_data = {
    0.0: {'cost': 702.44, 'emissions': 50.85},
    0.5: {'cost': 163.32, 'emissions': 60.97},
    1.0: {'cost': 139.66, 'emissions': 303.50}
}

for alpha in [0.0, 0.5, 1.0]:
    scenario_name = ['Emissions-Optimal', 'Balanced', 'Cost-Optimal'][int(alpha*2)]
    print(f"\nα={alpha} ({scenario_name}):")
    print("-" * 80)

    cost_base = baseline_data[alpha]['cost']
    cost_policy = nuclear_policy_data[alpha]['cost']
    cost_increase = cost_policy - cost_base
    cost_pct = (cost_increase / cost_base) * 100

    emis_base = baseline_data[alpha]['emissions']
    emis_policy = nuclear_policy_data[alpha]['emissions']
    emis_change = emis_policy - emis_base
    emis_pct = (emis_change / emis_base) * 100

    print(f"  Total Cost:  ${cost_base:>7.2f}B → ${cost_policy:>7.2f}B (+${cost_increase:>6.2f}B, +{cost_pct:>5.1f}%)")
    print(f"  Emissions:   {emis_base:>7.2f} MT → {emis_policy:>7.2f} MT ({emis_change:>+6.2f} MT, {emis_pct:>+5.1f}%)")

print("\n" + "="*100)
print("SUMMARY")
print("="*100)
print("\nThe 50% nuclear generation policy has dramatic impacts:")
print("\n1. Emissions-Optimal (α=0.0):")
print("   • Forces 24.8 GW of NEW NUCLEAR to be built (was 0 GW)")
print("   • Wind still dominates but reduced: 91.7 GW (was 98.5 GW)")
print("   • Cost increases massively: +$422.6B (+151%)")
print("   • Emissions barely change: +1.8 MT (+3.7%)")
print("   → Nuclear is VERY EXPENSIVE for emissions reduction!")

print("\n2. Balanced (α=0.5):")
print("   • Modest nuclear addition: 1.8 GW (was 0 GW)")
print("   • Wind reduced: 30.3 GW (was 34.9 GW)")
print("   • Cost increases: +$15.4B (+10.4%)")
print("   • Emissions increase slightly: +2.1 MT (+3.5%)")

print("\n3. Cost-Optimal (α=1.0):")
print("   • Small nuclear addition: 1.8 GW (was 0 GW)")
print("   • Wind increases: 1.3 GW (was 1.7 GW)")
print("   • Gas reduces: 18.5 GW (was 21.1 GW)")
print("   • Cost increases: +$17.5B (+14.3%)")
print("   • Emissions decrease: -12.3 MT (-3.9%)")
print("   → Gas is replaced by nuclear, slightly lowering emissions")

print("\n" + "="*100)

# Save comparison summary
summary_data = []
for alpha in [0.0, 0.5, 1.0]:
    scenario = ['Emissions-Optimal', 'Balanced', 'Cost-Optimal'][int(alpha*2)]
    summary_data.append({
        'Alpha': alpha,
        'Scenario': scenario,
        'Cost_Baseline_B': baseline_data[alpha]['cost'],
        'Cost_NuclearPolicy_B': nuclear_policy_data[alpha]['cost'],
        'Cost_Increase_B': nuclear_policy_data[alpha]['cost'] - baseline_data[alpha]['cost'],
        'Cost_Increase_Pct': ((nuclear_policy_data[alpha]['cost'] - baseline_data[alpha]['cost']) / baseline_data[alpha]['cost']) * 100,
        'Emissions_Baseline_MT': baseline_data[alpha]['emissions'],
        'Emissions_NuclearPolicy_MT': nuclear_policy_data[alpha]['emissions'],
        'Emissions_Change_MT': nuclear_policy_data[alpha]['emissions'] - baseline_data[alpha]['emissions'],
        'Emissions_Change_Pct': ((nuclear_policy_data[alpha]['emissions'] - baseline_data[alpha]['emissions']) / baseline_data[alpha]['emissions']) * 100,
        'Nuclear_Baseline_GW': baseline.loc['nuclear', f'α={alpha}'],
        'Nuclear_NuclearPolicy_GW': nuclear_policy.loc['nuclear', f'α={alpha}'],
        'Nuclear_Change_GW': nuclear_policy.loc['nuclear', f'α={alpha}'] - baseline.loc['nuclear', f'α={alpha}']
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('results/data/nuclear_policy_impact_summary.csv', index=False)
print("\n✓ Saved summary to results/data/nuclear_policy_impact_summary.csv")
