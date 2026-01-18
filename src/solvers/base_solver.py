"""Abstract base class for optimization solvers."""

from abc import ABC, abstractmethod


class Solver(ABC):
    """Abstract base class for optimization solvers."""

    @abstractmethod
    def solve(self, model, time_limit, mip_gap, tee):
        """Solve optimization model.

        Args:
            model: Pyomo ConcreteModel
            time_limit: Time limit in seconds
            mip_gap: MIP relative gap tolerance
            tee: Print solver output

        Returns:
            Solver result object
        """
        pass

    @abstractmethod
    def configure(self, options):
        """Configure solver-specific options.

        Args:
            options: Dict of solver options
        """
        pass
