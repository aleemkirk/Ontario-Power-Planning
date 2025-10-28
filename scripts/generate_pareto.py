"""
Script for generating Pareto frontier.

Usage:
    python scripts/generate_pareto.py --n-points 15 --solver highs
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.optimization.model import PowerSystemOptimization
from src.analysis.pareto import generate_pareto_frontier
from src.analysis.visualizations import plot_pareto_frontier


def main():
    parser = argparse.ArgumentParser(
        description='Generate Pareto frontier for multi-objective optimization'
    )
    parser.add_argument(
        '--n-points',
        type=int,
        default=15,
        help='Number of Pareto points to generate'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['weighted_sum', 'epsilon_constraint'],
        default='weighted_sum',
        help='Method for generating Pareto frontier'
    )
    parser.add_argument(
        '--solver',
        type=str,
        choices=['highs', 'gurobi', 'cplex'],
        default='highs',
        help='Solver to use'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/figures/pareto_frontier.png',
        help='Output file path for plot'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Pareto Frontier Generation")
    print("=" * 70)
    print(f"Number of points: {args.n_points}")
    print(f"Method: {args.method}")
    print(f"Solver: {args.solver}")
    print("=" * 70)

    # Create model
    print("\n[1/3] Building optimization model...")
    model = PowerSystemOptimization(
        start_year=2025,
        end_year=2045,
        data_path='data/processed/'
    )

    # Generate Pareto frontier
    print("\n[2/3] Generating Pareto frontier...")
    print(f"This may take several minutes ({args.n_points} optimization runs)...")

    try:
        pareto_solutions = generate_pareto_frontier(
            model=model,
            n_points=args.n_points,
            method=args.method,
            solver=args.solver
        )

        print(f"\n✓ Generated {len(pareto_solutions)} Pareto-optimal solutions")

        # Save results
        output_data = Path('results/data/pareto_solutions.csv')
        output_data.parent.mkdir(parents=True, exist_ok=True)
        pareto_solutions.to_csv(output_data, index=False)
        print(f"✓ Saved results to {output_data}")

        # Plot
        print("\n[3/3] Creating visualization...")
        output_fig = Path(args.output)
        output_fig.parent.mkdir(parents=True, exist_ok=True)
        plot_pareto_frontier(pareto_solutions, save_path=str(output_fig))
        print(f"✓ Saved plot to {output_fig}")

        print("\n" + "=" * 70)
        print("Pareto frontier generation completed!")
        print("=" * 70)

    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
