"""
HiGHS solver implementation.

HiGHS is an open-source linear programming solver, ideal for prototyping.
"""

from typing import Dict, Any
import pyomo.environ as pyo
from .base_solver import BaseSolver


class HiGHSSolver(BaseSolver):
    """HiGHS solver implementation."""

    def __init__(self, options: Dict[str, Any] = None):
        """
        Initialize HiGHS solver.

        Args:
            options: Solver options (time_limit, mip_gap, etc.)
        """
        super().__init__(options)
        self.setup()

    def setup(self):
        """Set up HiGHS solver with options."""
        try:
            self.solver = pyo.SolverFactory('appsi_highs')

            # Set common options
            if 'time_limit' in self.options:
                self.solver.options['time_limit'] = self.options['time_limit']
            if 'mip_gap' in self.options:
                self.solver.options['mip_rel_gap'] = self.options['mip_gap']

        except Exception as e:
            raise RuntimeError(f"Failed to initialize HiGHS solver: {e}")

    def solve(self, model: pyo.ConcreteModel):
        """
        Solve the optimization model with HiGHS.

        Args:
            model: Pyomo ConcreteModel to solve

        Returns:
            Solver results
        """
        if self.solver is None:
            raise RuntimeError("Solver not initialized. Call setup() first.")

        results = self.solver.solve(model, tee=True)
        return results
