"""
Pareto frontier generation for multi-objective optimization.

Implements methods:
- Weighted sum approach
- ε-constraint method
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from tqdm import tqdm


def generate_pareto_frontier(model_class, n_points: int = 15,
                            method: str = 'weighted_sum',
                            solver: str = 'highs',
                            **model_kwargs) -> pd.DataFrame:
    """
    Generate Pareto frontier showing cost-emissions trade-offs.

    Args:
        model_class: PowerSystemOptimization class (not instance!)
        n_points: Number of Pareto points to generate
        method: 'weighted_sum' or 'epsilon_constraint'
        solver: Solver to use
        **model_kwargs: Arguments to pass to model constructor

    Returns:
        DataFrame with Pareto-optimal solutions
    """
    print(f"\n{'=' * 70}")
    print(f"Generating Pareto Frontier ({n_points} points)")
    print(f"Method: {method}")
    print(f"{'=' * 70}\n")

    # Generate alpha values from 0 (emissions only) to 1 (cost only)
    alpha_values = np.linspace(0, 1, n_points)

    if method == 'weighted_sum':
        solutions = weighted_sum_method(model_class, alpha_values, solver, **model_kwargs)
    elif method == 'epsilon_constraint':
        solutions = epsilon_constraint_method(model_class, n_points, solver, **model_kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Convert to DataFrame
    df = pd.DataFrame(solutions)

    print(f"\n✓ Generated {len(df)} Pareto-optimal solutions")
    return df


def weighted_sum_method(model_class, alpha_values: np.ndarray, solver: str, **model_kwargs) -> List[Dict]:
    """
    Generate Pareto frontier using weighted sum method.

    Varies weight α from 0 to 1:
    - α = 0: minimize emissions only
    - α = 1: minimize cost only
    - 0 < α < 1: trade-off between objectives

    Args:
        model_class: PowerSystemOptimization class
        alpha_values: Array of weight values
        solver: Solver to use
        **model_kwargs: Arguments for model constructor

    Returns:
        List of solution dictionaries
    """
    solutions = []

    # First get normalization factors by running endpoints
    print("Step 1/3: Finding cost-optimal solution (for normalization)...")
    model_cost = model_class(**model_kwargs)
    results_cost = model_cost.optimize(objective='cost', solver=solver)

    print("Step 2/3: Finding emissions-optimal solution (for normalization)...")
    model_emis = model_class(**model_kwargs)
    results_emis = model_emis.optimize(objective='emissions', solver=solver)

    # Store normalization factors
    cost_max = results_cost['total_cost']
    emissions_max = results_cost['total_emissions']

    print(f"\nNormalization factors:")
    print(f"  Cost range: ${results_emis['total_cost']/1e9:.2f}B - ${cost_max/1e9:.2f}B")
    print(f"  Emissions range: {results_emis['total_emissions']/1e6:.2f} MT - {emissions_max/1e6:.2f} MT")

    print(f"\nStep 3/3: Generating {len(alpha_values)} Pareto points...")

    for i, alpha in enumerate(tqdm(alpha_values, desc="Optimizing")):
        # Create fresh model for each point
        model = model_class(**model_kwargs)
        model.model = None  # Force rebuild

        # Set normalization in model (will be used by multi_objective)
        # We'll pass this through build_model
        try:
            if alpha == 0:
                # Pure emissions minimization
                results = model.optimize(objective='emissions', solver=solver)
            elif alpha == 1:
                # Pure cost minimization
                results = model.optimize(objective='cost', solver=solver)
            else:
                # Multi-objective
                # Build model with normalization
                model.load_data()
                m = model.build_model(objective='multi', alpha=alpha)
                m.cost_normalization = cost_max
                m.emissions_normalization = emissions_max
                model.model = m

                # Solve
                results = model.optimize(objective='multi', solver=solver)

            if results:
                solutions.append({
                    'alpha': alpha,
                    'cost_billions': results['summary']['total_cost_billions'],
                    'emissions_megatons': results['summary']['total_emissions_megatons'],
                    'new_capacity_GW': results['summary']['total_new_capacity_GW'],
                    'final_capacity_GW': results['summary']['final_capacity_GW'],
                })
        except Exception as e:
            print(f"\n  Warning: Failed at alpha={alpha:.2f}: {e}")
            continue

    return solutions


def epsilon_constraint_method(model, n_points: int, solver: str) -> List[Dict]:
    """
    Generate Pareto frontier using ε-constraint method.

    Optimizes cost while constraining emissions to different levels.

    Args:
        model: PowerSystemOptimization instance
        n_points: Number of points to generate
        solver: Solver to use

    Returns:
        List of solution dictionaries
    """
    # TODO: Implement ε-constraint method
    pass


def normalize_objectives(solutions: List[Dict]) -> List[Dict]:
    """
    Normalize objective values for comparison.

    Args:
        solutions: List of solution dictionaries

    Returns:
        List of solutions with normalized objectives
    """
    # TODO: Implement normalization
    pass
