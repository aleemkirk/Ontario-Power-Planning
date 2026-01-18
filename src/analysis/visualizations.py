"""Visualization functions for optimization results."""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def plot_pareto_frontier(summary_df, output_dir='results/figures'):
    """Create Pareto frontier visualization.

    Args:
        summary_df: DataFrame with columns alpha, total_cost_billions, total_emissions_MT, etc.
        output_dir: Output directory for figure

    Returns:
        Path to saved figure
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Pareto Frontier: 20-Year Hourly Model (2025-2045)', fontsize=16, fontweight='bold')

    # 1. Main Pareto frontier
    ax1 = axes[0, 0]
    ax1.plot(summary_df['total_emissions_MT'], summary_df['total_cost_billions'],
             'o-', linewidth=2, markersize=8, color='#2E86AB')
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
    ax2.set_title('Cost Breakdown', fontsize=12, fontweight='bold')
    ax2.legend(['Capital', 'Operating', 'Maintenance'], loc='upper right')
    ax2.set_xticklabels([f'{a:.1f}' for a in summary_df['alpha']], rotation=45)

    # 3. Ramp penalty
    ax3 = axes[1, 0]
    ax3.plot(summary_df['alpha'], summary_df['ramp_penalty_millions'],
             's-', linewidth=2, markersize=6, color='#F77F00')
    ax3.set_xlabel('Alpha', fontsize=11)
    ax3.set_ylabel('Ramp Penalty (Million $)', fontsize=11)
    ax3.set_title('Ramp Violations', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. Solve time
    ax4 = axes[1, 1]
    ax4.bar(range(len(summary_df)), summary_df['solve_time_seconds'],
           color='#06A77D', alpha=0.7)
    ax4.set_xlabel('Pareto Point', fontsize=11)
    ax4.set_ylabel('Solve Time (s)', fontsize=11)
    ax4.set_title('Performance', fontsize=12, fontweight='bold')
    ax4.axhline(y=summary_df['solve_time_seconds'].mean(),
               color='red', linestyle='--', label=f'Mean: {summary_df["solve_time_seconds"].mean():.1f}s')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save
    fig_file = output_path / 'pareto_frontier_hourly.png'
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved plot to {fig_file}")
    return fig_file
