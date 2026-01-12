"""
Gurobi solver implementation.

Gurobi is a commercial solver with excellent performance for large-scale problems.
Free academic licenses available.
"""

from typing import Dict, Any
import pyomo.environ as pyo
from .base_solver import BaseSolver


class GurobiSolver(BaseSolver):
    """Gurobi solver implementation."""

    def __init__(self, options: Dict[str, Any] = None):
        """
        Initialize Gurobi solver.

        Args:
            options: Solver options (TimeLimit, MIPGap, etc.)
        """
        super().__init__(options)
        self.setup()

    def setup(self):
        """Set up Gurobi solver with options."""
        try:
            self.solver = pyo.SolverFactory('gurobi')

            # Set common options
            if 'time_limit' in self.options:
                self.solver.options['TimeLimit'] = self.options['time_limit']
            if 'mip_gap' in self.options:
                self.solver.options['MIPGap'] = self.options['mip_gap']
            if 'threads' in self.options:
                self.solver.options['Threads'] = self.options['threads']

        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gurobi solver: {e}. "
                             "Make sure Gurobi is installed and licensed.")

    def solve(self, model: pyo.ConcreteModel):
        """
        Solve the optimization model with Gurobi.

        Args:
            model: Pyomo ConcreteModel to solve

        Returns:
            Solver results
        """
        if self.solver is None:
            raise RuntimeError("Solver not initialized. Call setup() first.")

        results = self.solver.solve(model, tee=True)
        return results
