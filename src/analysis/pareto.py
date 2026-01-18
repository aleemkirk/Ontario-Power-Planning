"""Pareto frontier generation for multi-objective optimization."""

import time
import numpy as np
from pathlib import Path
import json
import pandas as pd


class ParetoFrontierGenerator:
    """Generate Pareto frontier for cost-emissions trade-offs."""

    def __init__(self, model_class, model_config):
        """Initialize generator.

        Args:
            model_class: PowerSystemOptimizationHourly class
            model_config: Dict with model configuration
        """
        self.model_class = model_class
        self.config = model_config

    def generate(self, n_points=11, alpha_range=(0.0, 1.0), solver='highs', time_limit=600):
        """Generate Pareto frontier.

        Args:
            n_points: Number of Pareto points
            alpha_range: (min, max) alpha values
            solver: Solver to use
            time_limit: Time limit per optimization (seconds)

        Returns:
            List of result dicts
        """
        alphas = np.linspace(alpha_range[0], alpha_range[1], n_points)
        results = []

        for i, alpha in enumerate(alphas):
            print(f"\n[{i+1}/{n_points}] α={alpha:.2f}")
            result = self._run_single(alpha, solver, time_limit)
            if result:
                results.append(result)

        return results

    def _run_single(self, alpha, solver, time_limit):
        """Run single optimization."""
        optimizer = self.model_class(**self.config)

        # Build model
        if alpha == 1.0:
            optimizer.build_model(objective='cost')
        elif alpha == 0.0:
            optimizer.build_model(objective='emissions')
        else:
            optimizer.build_model(objective='multi', alpha=alpha)

        # Solve
        start = time.time()
        result = optimizer.solve(solver=solver, time_limit=time_limit, tee=False, save_results=True)
        solve_time = time.time() - start

        if result['status'] != 'optimal':
            print(f"⚠ Status: {result['status']}")
            return None

        # Extract results
        res = optimizer.results
        end_year = self.config['end_year']

        return {
            'alpha': alpha,
            'status': result['status'],
            'solve_time': solve_time,
            'total_cost': res['total_cost'],
            'capex_cost': res['capex_cost'],
            'opex_cost': res['opex_cost'],
            'maintenance_cost': res['maintenance_cost'],
            'ramp_penalty_cost': res['ramp_penalty_cost'],
            'total_emissions': res['total_emissions'],
            'capacity_2045': res['capacity'][end_year],
            'capacity_2025': res['capacity'][self.config['start_year']]
        }

    def save_results(self, results, output_dir='results/data'):
        """Save results to CSV/JSON."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Summary CSV
        summary = pd.DataFrame([{
            'alpha': r['alpha'],
            'total_cost_billions': r['total_cost'] / 1e9,
            'total_emissions_MT': r['total_emissions'] / 1e6,
            'capex_cost_billions': r['capex_cost'] / 1e9,
            'opex_cost_billions': r['opex_cost'] / 1e9,
            'maintenance_cost_billions': r['maintenance_cost'] / 1e9,
            'ramp_penalty_millions': r['ramp_penalty_cost'] / 1e6,
            'solve_time_seconds': r['solve_time'],
            'status': r['status']
        } for r in results])
        summary.to_csv(output_path / 'pareto_frontier_hourly.csv', index=False)

        # Capacity CSV
        capacity = []
        for r in results:
            for plant, cap in r['capacity_2045'].items():
                capacity.append({
                    'alpha': r['alpha'],
                    'plant_type': plant,
                    'capacity_MW': cap,
                    'new_capacity_MW': cap - r['capacity_2025'][plant]
                })
        pd.DataFrame(capacity).to_csv(output_path / 'pareto_capacity_mix_hourly.csv', index=False)

        # Detailed JSON
        with open(output_path / 'pareto_frontier_hourly_detailed.json', 'w') as f:
            json.dump(results, f, indent=2)

        print(f"✓ Results saved to {output_dir}")
