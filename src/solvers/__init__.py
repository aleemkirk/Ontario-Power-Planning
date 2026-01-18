"""Solver abstraction layer for optimization."""

from .base_solver import Solver
from .highs_solver import HiGHSSolver
from .gurobi_solver import GurobiSolver


def get_solver(solver_type='highs'):
    """Factory function to get solver instance.

    Args:
        solver_type: 'highs' or 'gurobi'

    Returns:
        Solver instance
    """
    solvers = {
        'highs': HiGHSSolver,
        'gurobi': GurobiSolver
    }
    if solver_type not in solvers:
        raise ValueError(f"Unknown solver: {solver_type}")
    return solvers[solver_type]()


__all__ = ['Solver', 'HiGHSSolver', 'GurobiSolver', 'get_solver']
