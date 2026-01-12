"""
Abstract base class for optimization solvers.

Provides a common interface for different solvers (HiGHS, Gurobi, CPLEX).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import pyomo.environ as pyo


class BaseSolver(ABC):
    """Abstract base class for optimization solvers."""

    def __init__(self, options: Dict[str, Any] = None):
        """
        Initialize solver with options.

        Args:
            options: Solver-specific options dictionary
        """
        self.options = options or {}
        self.solver = None

    @abstractmethod
    def setup(self):
        """Set up the solver with configured options."""
        pass

    @abstractmethod
    def solve(self, model: pyo.ConcreteModel):
        """
        Solve the optimization model.

        Args:
            model: Pyomo ConcreteModel to solve

        Returns:
            Solver results object
        """
        pass

    def get_solver_name(self) -> str:
        """Return the solver name."""
        return self.__class__.__name__
