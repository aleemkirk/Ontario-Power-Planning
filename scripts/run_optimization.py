"""
Main execution script for running the optimization.

Usage:
    python scripts/run_optimization.py --objective cost --solver highs
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.optimization.model import PowerSystemOptimization
from data.load_data import DataLoader


def main():
    parser = argparse.ArgumentParser(
        description='Run Ontario Power Plant Optimization'
    )
    parser.add_argument(
        '--objective',
        type=str,
        choices=['cost', 'emissions', 'multi'],
        default='cost',
        help='Objective function to optimize'
    )
    parser.add_argument(
        '--solver',
        type=str,
        choices=['highs', 'gurobi', 'cplex'],
        default='highs',
        help='Solver to use'
    )
    parser.add_argument(
        '--start-year',
        type=int,
        default=2025,
        help='Start year of planning horizon'
    )
    parser.add_argument(
        '--end-year',
        type=int,
        default=2045,
        help='End year of planning horizon'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/processed/',
        help='Path to processed data files'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Ontario Power Plant Optimization")
    print("=" * 70)
    print(f"Objective: {args.objective}")
    print(f"Solver: {args.solver}")
    print(f"Planning horizon: {args.start_year}-{args.end_year}")
    print("=" * 70)

    # Validate data
    print("\n[1/3] Validating data...")
    loader = DataLoader(args.data_path)
    if not loader.validate_data():
        print("Error: Data validation failed")
        sys.exit(1)

    # Create model
    print("\n[2/3] Building optimization model...")
    model = PowerSystemOptimization(
        start_year=args.start_year,
        end_year=args.end_year,
        data_path=args.data_path
    )

    # Run optimization
    print("\n[3/3] Running optimization...")
    try:
        results = model.optimize(objective=args.objective, solver=args.solver)
        print("\n" + "=" * 70)
        print("Optimization completed successfully!")
        print("=" * 70)

        # TODO: Display summary results

    except Exception as e:
        print(f"\nError during optimization: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
