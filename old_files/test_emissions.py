"""
Test emissions minimization objective.

This finds the minimum emissions solution (one endpoint of Pareto frontier).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.optimization.model import PowerSystemOptimization

print("=" * 70)
print("Testing Emissions Minimization Objective")
print("=" * 70)

# Test both cost and emissions objectives
print("\n[1/2] Cost Minimization (for comparison)")
model_cost = PowerSystemOptimization(
    start_year=2025,
    end_year=2045,
    data_path='data/processed/',
    use_lead_times=True,
    use_retirements=True
)

results_cost = model_cost.optimize(objective='cost', solver='highs', time_limit=120)

print("\n[2/2] Emissions Minimization")
model_emissions = PowerSystemOptimization(
    start_year=2025,
    end_year=2045,
    data_path='data/processed/',
    use_lead_times=True,
    use_retirements=True
)

results_emissions = model_emissions.optimize(objective='emissions', solver='highs', time_limit=120)

if results_cost and results_emissions:
    print("\n" + "=" * 70)
    print("COMPARISON: Cost-Optimal vs Emissions-Optimal")
    print("=" * 70)

    # Extract summaries
    cost_summary = results_cost['summary']
    emis_summary = results_emissions['summary']

    print(f"\n{'Metric':<30} {'Cost-Optimal':>20} {'Emissions-Optimal':>20}")
    print("-" * 70)
    print(f"{'Total Cost (NPV)':<30} ${cost_summary['total_cost_billions']:>18.2f}B ${emis_summary['total_cost_billions']:>18.2f}B")
    print(f"{'Total Emissions':<30} {cost_summary['total_emissions_megatons']:>18.2f} MT {emis_summary['total_emissions_megatons']:>18.2f} MT")
    print(f"{'New Capacity Built':<30} {cost_summary['total_new_capacity_GW']:>18.2f} GW {emis_summary['total_new_capacity_GW']:>18.2f} GW")
    print(f"{'Final Capacity':<30} {cost_summary['final_capacity_GW']:>18.2f} GW {emis_summary['final_capacity_GW']:>18.2f} GW")

    # Calculate trade-offs
    cost_increase = emis_summary['total_cost_billions'] - cost_summary['total_cost_billions']
    emissions_reduction = cost_summary['total_emissions_megatons'] - emis_summary['total_emissions_megatons']

    print(f"\n{'Trade-off Analysis':<30}")
    print("-" * 70)
    print(f"{'Cost increase':<30} ${cost_increase:>18.2f}B ({(cost_increase/cost_summary['total_cost_billions']*100):+.1f}%)")
    print(f"{'Emissions reduction':<30} {emissions_reduction:>18.2f} MT ({(emissions_reduction/cost_summary['total_emissions_megatons']*100):.1f}%)")

    if emissions_reduction > 0:
        marginal_cost = (cost_increase * 1e9) / (emissions_reduction * 1e6)  # $/ton CO2
        print(f"{'Marginal abatement cost':<30} ${marginal_cost:>18.2f}/ton CO2")

    # Capacity mix comparison
    print(f"\n{'New Builds by Technology':<30}")
    print("-" * 70)

    builds_cost = results_cost['new_builds'].groupby('plant_type')['new_capacity_MW'].sum()
    builds_emis = results_emissions['new_builds'].groupby('plant_type')['new_capacity_MW'].sum()

    print(f"{'Plant Type':<15} {'Cost-Optimal (GW)':>20} {'Emissions-Optimal (GW)':>25}")
    print("-" * 70)
    for plant in ['nuclear', 'wind', 'solar', 'gas', 'hydro', 'biofuel']:
        cost_cap = builds_cost.get(plant, 0) / 1000
        emis_cap = builds_emis.get(plant, 0) / 1000
        if cost_cap > 0.01 or emis_cap > 0.01:
            print(f"{plant:<15} {cost_cap:>20.2f} {emis_cap:>25.2f}")

    # 2045 Generation mix
    print(f"\n{'2045 Generation Mix':<30}")
    print("-" * 70)

    gen_cost_2045 = results_cost['generation'][results_cost['generation']['year'] == 2045]
    gen_emis_2045 = results_emissions['generation'][results_emissions['generation']['year'] == 2045]

    total_gen_cost = gen_cost_2045['generation_MWh'].sum()
    total_gen_emis = gen_emis_2045['generation_MWh'].sum()

    print(f"{'Plant Type':<15} {'Cost-Optimal (%)':>20} {'Emissions-Optimal (%)':>25}")
    print("-" * 70)
    for plant in ['nuclear', 'wind', 'solar', 'gas', 'hydro', 'biofuel']:
        cost_pct = (gen_cost_2045[gen_cost_2045['plant_type'] == plant]['generation_MWh'].sum() / total_gen_cost * 100)
        emis_pct = (gen_emis_2045[gen_emis_2045['plant_type'] == plant]['generation_MWh'].sum() / total_gen_emis * 100)
        if cost_pct > 0.1 or emis_pct > 0.1:
            print(f"{plant:<15} {cost_pct:>19.1f}% {emis_pct:>24.1f}%")

    print("\n" + "=" * 70)
    print("Key Insights:")
    print("=" * 70)
    print(f"✓ Emissions can be reduced by {emissions_reduction:.0f} MT ({(emissions_reduction/cost_summary['total_emissions_megatons']*100):.1f}%)")
    print(f"✓ Cost of decarbonization: ${cost_increase:.2f}B additional investment")
    if emissions_reduction > 0:
        print(f"✓ Carbon abatement cost: ${marginal_cost:.0f}/ton CO2")
    print("✓ These are the two endpoints of the Pareto frontier")
