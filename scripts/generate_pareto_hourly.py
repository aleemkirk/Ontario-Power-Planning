"""Generate Pareto frontier for 20-year hourly model.

Usage:
    python generate_pareto_hourly.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.optimization.model_hourly import PowerSystemOptimizationHourly
from src.analysis import ParetoFrontierGenerator, plot_pareto_frontier
import pandas as pd


if __name__ == '__main__':
    # Model configuration
    config = {
        'start_year': 2025,
        'end_year': 2045,
        'n_rep_days': 12,
        'use_soft_ramp_constraints': True,
        'ramp_penalty': 1000.0,
        'use_lead_times': True,
        'use_retirements': False
    }

    # Generate Pareto frontier
    print("="*80)
    print("PARETO FRONTIER GENERATION")
    print("="*80)

    generator = ParetoFrontierGenerator(PowerSystemOptimizationHourly, config)
    results = generator.generate(n_points=11, solver='highs', time_limit=600)

    # Save results
    generator.save_results(results, output_dir='results/data')

    # Create visualization
    summary_df = pd.read_csv('results/data/pareto_frontier_hourly.csv')
    plot_pareto_frontier(summary_df, output_dir='results/figures')

    print(f"\n✓ Generated {len(results)} Pareto points")
    print(f"✓ Results saved to results/data/")
    print(f"✓ Visualization saved to results/figures/")
