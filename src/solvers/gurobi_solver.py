"""Gurobi solver implementation."""

import pyomo.environ as pyo
from .base_solver import Solver


class GurobiSolver(Solver):
    """Gurobi commercial solver."""

    def solve(self, model, time_limit=600, mip_gap=0.01, tee=False):
        """Solve with Gurobi."""
        opt = pyo.SolverFactory('gurobi')
        opt.options['TimeLimit'] = time_limit
        opt.options['MIPGap'] = mip_gap
        logfile = 'gurobi.log' if tee else None
        return opt.solve(model, tee=tee, logfile=logfile)

    def configure(self, options):
        """Configure Gurobi options."""
        pass  # Options set in solve()
