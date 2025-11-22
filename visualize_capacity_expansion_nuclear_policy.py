"""
Visualize capacity expansion timelines with 50% nuclear generation policy.

Creates graphs showing when to build each type of power plant over 20 years
for three scenarios, WITH the constraint that nuclear must account for at least
50% of all power generated in any given year.

Usage:
    python visualize_capacity_expansion_nuclear_policy.py
"""

import sys
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.optimization.model_hourly import PowerSystemOptimizationHourly


def run_and_extract_builds(alpha, start_year=2025, end_year=2045):
    """
    Run optimization for given alpha WITH NUCLEAR POLICY and extract detailed build schedule.

    Args:
        alpha: Weight for cost objective (0 to 1)
        start_year: First year of planning horizon
        end_year: Last year of planning horizon

    Returns:
        DataFrame with new builds by year and plant type
    """
    print(f"\n{'='*80}")
    print(f"Running optimization WITH NUCLEAR POLICY: α = {alpha}")
    print(f"Policy: Nuclear must be ≥50% of generation each year")
    print(f"{'='*80}")

    # Initialize model WITH NUCLEAR POLICY
    optimizer = PowerSystemOptimizationHourly(
        start_year=start_year,
        end_year=end_year,
        n_rep_days=12,
        use_soft_ramp_constraints=True,
        ramp_penalty=1000.0,
        use_lead_times=True,
        use_retirements=False,
        use_nuclear_policy=True,        # ENABLE NUCLEAR POLICY
        min_nuclear_share=0.5           # 50% minimum nuclear generation
    )

    # Build model
    if alpha == 1.0:
        optimizer.build_model(objective='cost')
    elif alpha == 0.0:
        optimizer.build_model(objective='emissions')
    else:
        optimizer.build_model(objective='multi', alpha=alpha)

    # Solve
    result = optimizer.solve(solver='highs', time_limit=600, tee=False, save_results=True)

    if result['status'] != 'optimal':
        print(f"⚠️  WARNING: Solution status = {result['status']}")
        return None

    # Extract results
    results = optimizer.results
    new_builds = results['new_builds']

    print(f"✓ Solution found")
    print(f"  Total cost: ${results['total_cost']/1e9:.2f}B")
    print(f"  Total emissions: {results['total_emissions']/1e6:.2f} MT")
    print(f"  New capacity: {new_builds['new_capacity_MW'].sum()/1000:.2f} GW")

    # Show capacity mix to verify nuclear share
    capacity_2045 = results['capacity'][end_year]
    print(f"  Final capacity (2045):")
    for plant_type in ['nuclear', 'wind', 'gas', 'hydro', 'solar', 'biofuel']:
        print(f"    {plant_type}: {capacity_2045[plant_type]/1000:.2f} GW")

    return new_builds


def visualize_expansion_timeline(builds_data, alpha_values, output_dir='results/figures'):
    """
    Create capacity expansion timeline visualizations.

    Args:
        builds_data: Dictionary mapping alpha -> DataFrame of new builds
        alpha_values: List of alpha values to visualize
        output_dir: Directory to save figures
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Define colors for each plant type
    plant_colors = {
        'nuclear': '#E63946',    # Red
        'wind': '#06A77D',       # Green
        'solar': '#F77F00',      # Orange
        'gas': '#457B9D',        # Blue
        'hydro': '#1D3557',      # Dark blue
        'biofuel': '#A8DADC'     # Light blue
    }

    # Create figure with 3 subplots (one for each alpha)
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle('Capacity Expansion with 50% Nuclear Policy: When to Build (2025-2045)',
                 fontsize=16, fontweight='bold', y=0.995)

    scenario_names = {
        0.0: 'Emissions-Optimal (α=0.0) + 50% Nuclear',
        0.5: 'Balanced (α=0.5) + 50% Nuclear',
        1.0: 'Cost-Optimal (α=1.0) + 50% Nuclear'
    }

    for idx, alpha in enumerate(alpha_values):
        ax = axes[idx]
        builds = builds_data[alpha]

        # Pivot data for stacked bar chart
        pivot_data = builds.groupby(['year', 'plant_type'])['new_capacity_MW'].sum().unstack(fill_value=0)

        # Reorder columns to match plant_colors order
        plant_types_order = ['nuclear', 'hydro', 'gas', 'wind', 'solar', 'biofuel']
        pivot_data = pivot_data.reindex(columns=plant_types_order, fill_value=0)

        # Create stacked bar chart
        pivot_data.plot(kind='bar', stacked=True, ax=ax,
                       color=[plant_colors[pt] for pt in pivot_data.columns],
                       width=0.8, edgecolor='white', linewidth=0.5)

        # Formatting
        ax.set_title(scenario_names[alpha], fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel('Year', fontsize=11)
        ax.set_ylabel('New Capacity Added (MW)', fontsize=11)
        ax.legend(title='Plant Type', bbox_to_anchor=(1.01, 1), loc='upper left',
                 frameon=True, fancybox=True, shadow=True)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        # Rotate x-axis labels
        ax.set_xticklabels(pivot_data.index, rotation=45, ha='right')

        # Add total new capacity annotation
        total_new = builds['new_capacity_MW'].sum()
        ax.text(0.02, 0.98, f'Total New: {total_new/1000:.1f} GW\n50% Nuclear Policy',
               transform=ax.transAxes, fontsize=10, fontweight='bold',
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

        # Highlight nuclear builds specifically
        if 'nuclear' in pivot_data.columns:
            nuclear_total = pivot_data['nuclear'].sum()
            if nuclear_total > 0:
                ax.text(0.98, 0.98, f'Nuclear: {nuclear_total/1000:.1f} GW',
                       transform=ax.transAxes, fontsize=10, fontweight='bold',
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='#FFE5E5', alpha=0.7))

    plt.tight_layout()

    # Save figure
    fig_file = output_path / 'capacity_expansion_timeline_nuclear_policy.png'
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved expansion timeline to {fig_file}")

    plt.close()

    return fig_file


def create_cumulative_capacity_plot(builds_data, alpha_values, output_dir='results/figures'):
    """
    Create cumulative capacity plots showing total capacity over time.
    """
    output_path = Path(output_dir)

    plant_colors = {
        'nuclear': '#E63946',
        'wind': '#06A77D',
        'solar': '#F77F00',
        'gas': '#457B9D',
        'hydro': '#1D3557',
        'biofuel': '#A8DADC'
    }

    initial_capacity = {
        'nuclear': 13000,
        'wind': 5575,
        'solar': 2669,
        'gas': 10500,
        'hydro': 8500,
        'biofuel': 205
    }

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle('Cumulative Capacity Evolution with 50% Nuclear Policy (2025-2045)',
                 fontsize=16, fontweight='bold', y=0.995)

    scenario_names = {
        0.0: 'Emissions-Optimal (α=0.0) + 50% Nuclear',
        0.5: 'Balanced (α=0.5) + 50% Nuclear',
        1.0: 'Cost-Optimal (α=1.0) + 50% Nuclear'
    }

    for idx, alpha in enumerate(alpha_values):
        ax = axes[idx]
        builds = builds_data[alpha]

        years = sorted(builds['year'].unique())
        cumulative = {}

        for plant_type in initial_capacity.keys():
            cumulative[plant_type] = []
            total = initial_capacity[plant_type]

            for year in years:
                year_builds = builds[(builds['year'] == year) &
                                    (builds['plant_type'] == plant_type)]['new_capacity_MW'].sum()
                total += year_builds
                cumulative[plant_type].append(total)

        plant_types_order = ['nuclear', 'hydro', 'gas', 'wind', 'solar', 'biofuel']
        cumulative_df = pd.DataFrame(cumulative, index=years)
        cumulative_df = cumulative_df[plant_types_order]

        cumulative_df.plot(kind='area', stacked=True, ax=ax,
                          color=[plant_colors[pt] for pt in cumulative_df.columns],
                          alpha=0.7, linewidth=0)

        ax.set_title(scenario_names[alpha], fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel('Year', fontsize=11)
        ax.set_ylabel('Total Capacity (MW)', fontsize=11)
        ax.legend(title='Plant Type', bbox_to_anchor=(1.01, 1), loc='upper left',
                 frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        final_total = cumulative_df.iloc[-1].sum()
        nuclear_final = cumulative_df.iloc[-1]['nuclear']
        ax.text(0.02, 0.98, f'Final: {final_total/1000:.1f} GW\nNuclear: {nuclear_final/1000:.1f} GW',
               transform=ax.transAxes, fontsize=10, fontweight='bold',
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    plt.tight_layout()

    fig_file = output_path / 'cumulative_capacity_evolution_nuclear_policy.png'
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved cumulative capacity plot to {fig_file}")

    plt.close()

    return fig_file


def create_build_comparison_table(builds_data, alpha_values):
    """
    Create comparison table showing total new builds by plant type.
    """
    comparison = {}

    for alpha in alpha_values:
        builds = builds_data[alpha]
        totals = builds.groupby('plant_type')['new_capacity_MW'].sum()
        comparison[alpha] = totals

    comparison_df = pd.DataFrame(comparison)
    comparison_df.columns = [f'α={a}' for a in comparison_df.columns]
    comparison_df = comparison_df.fillna(0)

    # Add total row
    comparison_df.loc['TOTAL'] = comparison_df.sum()

    # Convert to GW
    comparison_df = comparison_df / 1000

    print("\n" + "="*80)
    print("NEW CAPACITY BUILT BY PLANT TYPE (GW) - WITH 50% NUCLEAR POLICY")
    print("="*80)
    print(comparison_df.to_string(float_format=lambda x: f'{x:>8.2f}'))
    print("="*80)

    # Save to CSV
    output_file = Path('results/data') / 'capacity_builds_comparison_nuclear_policy.csv'
    comparison_df.to_csv(output_file)
    print(f"\n✓ Saved comparison table to {output_file}")

    return comparison_df


if __name__ == '__main__':
    print("="*80)
    print("CAPACITY EXPANSION WITH 50% NUCLEAR GENERATION POLICY")
    print("="*80)
    print("\nPolicy Constraint: Nuclear must account for ≥50% of generation each year")
    print("\nGenerating visualizations for 3 scenarios:")
    print("  1. Emissions-Optimal (α=0.0) - Minimize emissions")
    print("  2. Balanced (α=0.5) - Balance cost and emissions")
    print("  3. Cost-Optimal (α=1.0) - Minimize cost")

    alpha_values = [0.0, 0.5, 1.0]

    # Run optimizations and collect build data
    builds_data = {}

    for alpha in alpha_values:
        builds = run_and_extract_builds(alpha, start_year=2025, end_year=2045)
        if builds is not None:
            builds_data[alpha] = builds

    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)

    # Create timeline visualization
    visualize_expansion_timeline(builds_data, alpha_values)

    # Create cumulative capacity plot
    create_cumulative_capacity_plot(builds_data, alpha_values)

    # Create comparison table
    create_build_comparison_table(builds_data, alpha_values)

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  • results/figures/capacity_expansion_timeline_nuclear_policy.png")
    print("  • results/figures/cumulative_capacity_evolution_nuclear_policy.png")
    print("  • results/data/capacity_builds_comparison_nuclear_policy.csv")
    print("\n" + "="*80)
