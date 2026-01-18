"""HiGHS solver implementation."""

import pyomo.environ as pyo
from .base_solver import Solver


class HiGHSSolver(Solver):
    """HiGHS open-source solver."""

    def solve(self, model, time_limit=600, mip_gap=0.01, tee=False):
        """Solve with HiGHS."""
        opt = pyo.SolverFactory('appsi_highs')
        opt.options['time_limit'] = time_limit
        opt.options['mip_rel_gap'] = mip_gap
        return opt.solve(model, tee=tee)

    def configure(self, options):
        """Configure HiGHS options."""
        pass  # Options set in solve()
