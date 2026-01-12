"""
Generate Pareto frontier for 20-year hourly model.

Runs multi-objective optimization with different weight values (alpha)
to generate the cost-emissions trade-off curve.

Alpha values:
- α = 1.0: Minimize cost only
- α = 0.0: Minimize emissions only
- α ∈ (0, 1): Trade-off between objectives

Usage:
    python generate_pareto_hourly.py
"""

import time
import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.optimization.model_hourly import PowerSystemOptimizationHourly
import pyomo.environ as pyo


def run_single_optimization(alpha, start_year=2025, end_year=2045):
    """
    Run a single optimization with given alpha weight.

    Args:
        alpha: Weight for cost objective (0 to 1)
        start_year: First year of planning horizon
        end_year: Last year of planning horizon

    Returns:
        Dictionary with results
    """
    print(f"\n{'='*80}")
    print(f"RUNNING OPTIMIZATION: α = {alpha:.2f}")
    print(f"{'='*80}")

    if alpha == 1.0:
        print("Objective: MINIMIZE COST ONLY")
    elif alpha == 0.0:
        print("Objective: MINIMIZE EMISSIONS ONLY")
    else:
        print(f"Objective: {alpha*100:.0f}% cost, {(1-alpha)*100:.0f}% emissions")

    # Initialize model
    optimizer = PowerSystemOptimizationHourly(
        start_year=start_year,
        end_year=end_year,
        n_rep_days=12,
        use_soft_ramp_constraints=True,
        ramp_penalty=1000.0,
        use_lead_times=True,
        use_retirements=False
    )

    # Build model with appropriate objective
    start_build = time.time()

    if alpha == 1.0:
        optimizer.build_model(objective='cost')
    elif alpha == 0.0:
        optimizer.build_model(objective='emissions')
    else:
        optimizer.build_model(objective='multi', alpha=alpha)

    build_time = time.time() - start_build

    # Solve
    start_solve = time.time()
    result = optimizer.solve(solver='highs', time_limit=600, tee=False, save_results=True)
    solve_time = time.time() - start_solve

    if result['status'] != 'optimal':
        print(f"⚠️  WARNING: Solution status = {result['status']}")
        return None

    # Extract results
    results = optimizer.results

    # Get capacity mix for final year
    capacity_2045 = results['capacity'][end_year]

    print(f"\n[RESULTS]")
    print(f"  Status: {result['status']}")
    print(f"  Solve time: {solve_time:.1f}s")
    print(f"  Total cost: ${results['total_cost']/1e9:.2f}B")
    print(f"  Total emissions: {results['total_emissions']/1e6:.2f} MT")
    print(f"  Ramp penalty: ${results['ramp_penalty_cost']/1e6:.2f}M")
    print(f"  Final capacity: {sum(capacity_2045.values()):,.0f} MW")

    return {
        'alpha': alpha,
        'status': result['status'],
        'solve_time': solve_time,
        'build_time': build_time,
        'total_cost': results['total_cost'],
        'capex_cost': results['capex_cost'],
        'opex_cost': results['opex_cost'],
        'maintenance_cost': results['maintenance_cost'],
        'ramp_penalty_cost': results['ramp_penalty_cost'],
        'total_emissions': results['total_emissions'],
        'new_builds': results['new_builds'],
        'capacity_2025': results['capacity'][start_year],
        'capacity_2045': capacity_2045,
        'generation': results['generation']
    }


def generate_pareto_frontier(n_points=11, start_year=2025, end_year=2045):
    """
    Generate Pareto frontier with n_points solutions.

    Args:
        n_points: Number of Pareto points (default 11: α = 0.0, 0.1, ..., 1.0)
        start_year: First year of planning horizon
        end_year: Last year of planning horizon

    Returns:
        List of result dictionaries
    """
    print("="*80)
    print("PARETO FRONTIER GENERATION - 20-YEAR HOURLY MODEL")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Planning horizon: {start_year}-{end_year}")
    print(f"  Number of points: {n_points}")
    print(f"  Representative days: 12")
    print(f"  Ramp constraints: ACTIVATED")
    print(f"  Lead times: ENABLED")
    print(f"  Retirements: DISABLED")

    # Generate alpha values
    alpha_values = np.linspace(0.0, 1.0, n_points)

    print(f"\nAlpha values: {', '.join([f'{a:.2f}' for a in alpha_values])}")

    # Store all results
    pareto_results = []

    # Run optimizations
    for i, alpha in enumerate(alpha_values):
        print(f"\n[Progress: {i+1}/{n_points}]")

        result = run_single_optimization(alpha, start_year, end_year)

        if result is None:
            print(f"⚠️  Skipping α={alpha:.2f} due to solve failure")
            continue

        pareto_results.append(result)

    print(f"\n{'='*80}")
    print(f"PARETO FRONTIER GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Successfully generated {len(pareto_results)}/{n_points} solutions")

    return pareto_results


def save_pareto_results(pareto_results, output_dir='results/data'):
    """Save Pareto frontier results to files."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n[Saving Results]")

    # Create summary DataFrame
    summary_data = []
    for result in pareto_results:
        summary_data.append({
            'alpha': result['alpha'],
            'total_cost_billions': result['total_cost'] / 1e9,
            'total_emissions_MT': result['total_emissions'] / 1e6,
            'capex_cost_billions': result['capex_cost'] / 1e9,
            'opex_cost_billions': result['opex_cost'] / 1e9,
            'maintenance_cost_billions': result['maintenance_cost'] / 1e9,
            'ramp_penalty_millions': result['ramp_penalty_cost'] / 1e6,
            'solve_time_seconds': result['solve_time'],
            'status': result['status']
        })

    summary_df = pd.DataFrame(summary_data)

    # Save summary
    summary_file = output_path / 'pareto_frontier_hourly.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"✓ Saved Pareto frontier summary to {summary_file}")

    # Save capacity mix for each solution
    capacity_data = []
    for result in pareto_results:
        for plant_type, capacity in result['capacity_2045'].items():
            capacity_data.append({
                'alpha': result['alpha'],
                'plant_type': plant_type,
                'capacity_MW': capacity,
                'new_capacity_MW': capacity - result['capacity_2025'][plant_type]
            })

    capacity_df = pd.DataFrame(capacity_data)
    capacity_file = output_path / 'pareto_capacity_mix_hourly.csv'
    capacity_df.to_csv(capacity_file, index=False)
    print(f"✓ Saved capacity mix to {capacity_file}")

    # Save detailed results as JSON
    json_file = output_path / 'pareto_frontier_hourly_detailed.json'

    # Convert DataFrames to dicts for JSON serialization
    json_data = []
    for result in pareto_results:
        json_data.append({
            'alpha': result['alpha'],
            'status': result['status'],
            'solve_time': result['solve_time'],
            'total_cost': result['total_cost'],
            'total_emissions': result['total_emissions'],
            'ramp_penalty_cost': result['ramp_penalty_cost'],
            'capacity_2045': result['capacity_2045'],
            'capacity_2025': result['capacity_2025']
        })

    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"✓ Saved detailed results to {json_file}")

    return summary_df, capacity_df


def visualize_pareto_frontier(summary_df, output_dir='results/figures'):
    """Create Pareto frontier visualization."""

    import matplotlib.pyplot as plt
    import seaborn as sns

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n[Creating Visualizations]")

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Pareto Frontier: 20-Year Hourly Model (2025-2045)', fontsize=16, fontweight='bold')

    # 1. Main Pareto frontier
    ax1 = axes[0, 0]
    ax1.plot(summary_df['total_emissions_MT'], summary_df['total_cost_billions'],
             'o-', linewidth=2, markersize=8, color='#2E86AB', label='Pareto frontier')
    ax1.set_xlabel('Total Emissions (MT CO₂)', fontsize=11)
    ax1.set_ylabel('Total Cost (Billion $)', fontsize=11)
    ax1.set_title('Cost vs Emissions Trade-off', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Annotate endpoints
    cost_min = summary_df.loc[summary_df['alpha'] == 1.0].iloc[0]
    emissions_min = summary_df.loc[summary_df['alpha'] == 0.0].iloc[0]

    ax1.annotate('Cost-optimal\n(α=1.0)',
                xy=(cost_min['total_emissions_MT'], cost_min['total_cost_billions']),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    ax1.annotate('Emissions-optimal\n(α=0.0)',
                xy=(emissions_min['total_emissions_MT'], emissions_min['total_cost_billions']),
                xytext=(10, -20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    # 2. Cost breakdown
    ax2 = axes[0, 1]
    cost_components = summary_df[['alpha', 'capex_cost_billions', 'opex_cost_billions',
                                   'maintenance_cost_billions']].copy()
    cost_components = cost_components.set_index('alpha')
    cost_components.plot(kind='bar', stacked=True, ax=ax2,
                        color=['#E63946', '#F1FAEE', '#A8DADC'])
    ax2.set_xlabel('Alpha (cost weight)', fontsize=11)
    ax2.set_ylabel('Cost (Billion $)', fontsize=11)
    ax2.set_title('Cost Breakdown by Objective Weight', fontsize=12, fontweight='bold')
    ax2.legend(['Capital', 'Operating', 'Maintenance'], loc='upper right')
    ax2.set_xticklabels([f'{a:.1f}' for a in summary_df['alpha']], rotation=45)

    # 3. Ramp penalty vs alpha
    ax3 = axes[1, 0]
    ax3.plot(summary_df['alpha'], summary_df['ramp_penalty_millions'],
             's-', linewidth=2, markersize=6, color='#F77F00')
    ax3.set_xlabel('Alpha (cost weight)', fontsize=11)
    ax3.set_ylabel('Ramp Penalty (Million $)', fontsize=11)
    ax3.set_title('Ramp Constraint Violations by Objective', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # 4. Solve time
    ax4 = axes[1, 1]
    ax4.bar(range(len(summary_df)), summary_df['solve_time_seconds'],
           color='#06A77D', alpha=0.7)
    ax4.set_xlabel('Pareto Point Index', fontsize=11)
    ax4.set_ylabel('Solve Time (seconds)', fontsize=11)
    ax4.set_title('Computational Performance', fontsize=12, fontweight='bold')
    ax4.axhline(y=summary_df['solve_time_seconds'].mean(),
               color='red', linestyle='--', label=f'Mean: {summary_df["solve_time_seconds"].mean():.1f}s')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save figure
    fig_file = output_path / 'pareto_frontier_hourly.png'
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Pareto frontier plot to {fig_file}")

    plt.close()

    return fig_file


if __name__ == '__main__':
    # Generate Pareto frontier with 11 points
    start_time = time.time()

    pareto_results = generate_pareto_frontier(n_points=11, start_year=2025, end_year=2045)

    total_time = time.time() - start_time

    # Save results
    summary_df, capacity_df = save_pareto_results(pareto_results)

    # Create visualizations
    visualize_pareto_frontier(summary_df)

    # Print final summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"\nTotal execution time: {total_time/60:.2f} minutes")
    print(f"Solutions generated: {len(pareto_results)}")
    print(f"Average solve time: {summary_df['solve_time_seconds'].mean():.1f}s")

    print(f"\nPareto Frontier Range:")
    print(f"  Cost:      ${summary_df['total_cost_billions'].min():.2f}B - ${summary_df['total_cost_billions'].max():.2f}B")
    print(f"  Emissions: {summary_df['total_emissions_MT'].min():.2f} MT - {summary_df['total_emissions_MT'].max():.2f} MT")

    print(f"\nTrade-off Analysis:")
    cost_range = summary_df['total_cost_billions'].max() - summary_df['total_cost_billions'].min()
    emissions_range = summary_df['total_emissions_MT'].max() - summary_df['total_emissions_MT'].min()
    print(f"  Cost range: ${cost_range:.2f}B ({cost_range/summary_df['total_cost_billions'].min()*100:.1f}%)")
    print(f"  Emissions range: {emissions_range:.2f} MT ({emissions_range/summary_df['total_emissions_MT'].min()*100:.1f}%)")

    # Calculate marginal cost of carbon reduction
    if len(summary_df) > 1:
        marginal_costs = []
        for i in range(len(summary_df) - 1):
            delta_cost = (summary_df.iloc[i+1]['total_cost_billions'] - summary_df.iloc[i]['total_cost_billions']) * 1e9
            delta_emissions = (summary_df.iloc[i+1]['total_emissions_MT'] - summary_df.iloc[i]['total_emissions_MT']) * 1e6
            if delta_emissions < 0:  # Emissions reduction
                marginal_cost = -delta_cost / delta_emissions  # $/ton
                marginal_costs.append(marginal_cost)

        if marginal_costs:
            avg_marginal_cost = np.mean(marginal_costs)
            print(f"\nMarginal Cost of Carbon Reduction:")
            print(f"  Average: ${avg_marginal_cost:.2f}/ton CO₂")
            print(f"  Range: ${min(marginal_costs):.2f} - ${max(marginal_costs):.2f}/ton CO₂")

    print(f"\n✓ All results saved to results/data/")
    print(f"✓ Visualization saved to results/figures/")
    print(f"\n{'='*80}")
