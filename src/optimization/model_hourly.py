"""
Hourly resolution optimization model for power system planning.

Extends the base PowerSystemOptimization model to include hourly dispatch
with representative days, enabling proper modeling of ramp rate constraints
and operational flexibility.

Key enhancements over annual model:
- Hourly dispatch variables p_hourly[t,d,h,i]
- Ramp rate constraints (ACTIVATED)
- Representative day weighting
- Operational flexibility properly valued
"""

import pyomo.environ as pyo
from typing import Dict, Optional
import pandas as pd
import numpy as np
from pathlib import Path

from .variables import define_variables
from .objectives import total_cost_objective_hourly, total_emissions_objective_hourly
from .constraints_hourly import add_hourly_constraints
from .constraints import (
    reserve_margin_constraint,
    capacity_dynamics_constraint
)
from .constraints_nuclear_policy import add_nuclear_policy_constraints
from ..utils.financial import calculate_discount_factor
from ..utils.load_hourly_data import setup_hourly_data


class PowerSystemOptimizationHourly:
    """
    Hourly resolution optimization model for Ontario power system planning.

    Uses representative days to reduce computational complexity while
    maintaining operational realism (ramp rates, hourly dispatch).

    Key differences from annual model:
    - Decision variables: p_hourly[t,d,h,i] instead of p[t,i]
    - Constraints: Hourly demand satisfaction, ramp rate constraints
    - Time resolution: 12 rep days × 24 hours = 288 hourly constraints per year
    """

    def __init__(self,
                 start_year: int = 2025,
                 end_year: int = 2045,
                 data_path: str = 'data/processed/',
                 discount_rate: float = 0.0392,
                 use_lead_times: bool = True,
                 use_retirements: bool = True,
                 n_rep_days: int = 12,
                 use_soft_ramp_constraints: bool = True,
                 ramp_penalty: float = 1000.0,
                 use_nuclear_policy: bool = False,
                 min_nuclear_share: float = 0.5):
        """
        Initialize hourly resolution optimization model.

        Args:
            start_year: First year of planning horizon
            end_year: Last year of planning horizon
            data_path: Path to processed data files
            discount_rate: Real discount rate (default 3.92%)
            use_lead_times: Include construction lead times
            use_retirements: Include plant retirements
            n_rep_days: Number of representative days (default 12)
            use_soft_ramp_constraints: Use soft constraints with penalty (default True)
            ramp_penalty: Penalty for ramp violations ($/MW, default $1000)
            use_nuclear_policy: Enforce minimum nuclear generation share (default False)
            min_nuclear_share: Minimum nuclear share if policy enabled (default 0.5 = 50%)
        """
        self.start_year = start_year
        self.end_year = end_year
        self.data_path = Path(data_path)
        self.discount_rate = discount_rate
        self.use_lead_times = use_lead_times
        self.use_retirements = use_retirements
        self.n_rep_days = n_rep_days
        self.use_soft_ramp_constraints = use_soft_ramp_constraints
        self.ramp_penalty = ramp_penalty
        self.use_nuclear_policy = use_nuclear_policy
        self.min_nuclear_share = min_nuclear_share

        self.model = None
        self.results = None
        self.solver_results = None

        # Data containers
        self.plant_params = None
        self.demand_data = None
        self.initial_cap = None
        self.hourly_data = None

        print(f"\n[Hourly Resolution Model Initialized]")
        print(f"Planning horizon: {start_year}-{end_year} ({end_year-start_year+1} years)")
        print(f"Representative days: {n_rep_days}")
        print(f"Ramp constraints: {'Soft' if use_soft_ramp_constraints else 'Hard'}")

    def load_data(self):
        """
        Load all required data including hourly demand profiles.
        """
        import json

        print("\n[Loading Data]")

        # Load plant parameters
        with open(self.data_path / 'plant_parameters.json', 'r') as f:
            self.plant_params = json.load(f)
        print(f"✓ Plant parameters loaded")

        # Load demand forecast (annual totals for validation)
        self.demand_data = pd.read_csv(self.data_path / 'demand_forecast.csv')
        self.demand_data = self.demand_data[
            (self.demand_data['year'] >= self.start_year) &
            (self.demand_data['year'] <= self.end_year)
        ].reset_index(drop=True)
        print(f"✓ Annual demand forecast: {len(self.demand_data)} years")

        # Load initial capacity
        with open(self.data_path / 'initial_capacity.json', 'r') as f:
            self.initial_cap = json.load(f)
        print(f"✓ Initial capacity (2025): {sum(self.initial_cap.values()):.0f} MW")

        # Load hourly demand data (representative days scaled for all years)
        print(f"✓ Loading hourly demand data...")
        self.hourly_data = setup_hourly_data(
            start_year=self.start_year,
            end_year=self.end_year,
            force_regenerate=False
        )
        print(f"✓ Hourly data: {self.hourly_data['n_rep_days']} rep days × 24 hours")

    def calculate_retirements(self):
        """Calculate plant retirements (same as annual model)."""
        retirements = {}
        for year in range(self.start_year, self.end_year + 1):
            for plant_type in self.plant_params['capex'].keys():
                lifespan = self.plant_params['lifespan'][plant_type]
                initial_cap = self.initial_cap[plant_type]
                annual_retirement = initial_cap / lifespan
                years_elapsed = year - self.start_year
                if years_elapsed < lifespan:
                    retirements[(year, plant_type)] = annual_retirement
                else:
                    retirements[(year, plant_type)] = 0.0
        return retirements

    def build_model(self, objective: str = 'cost', alpha: float = 0.5):
        """
        Build the Pyomo hourly optimization model.

        Args:
            objective: 'cost', 'emissions', or 'multi'
            alpha: Weight for multi-objective (only used if objective='multi')
        """
        if self.plant_params is None:
            self.load_data()

        print(f"\n[Building Hourly Model]")
        print(f"Objective: {objective}")
        if self.use_lead_times:
            print(f"✓ Construction lead times enabled")
        if self.use_retirements:
            print(f"✓ Plant retirements enabled")

        # Create concrete model
        m = pyo.ConcreteModel(name="Ontario Power Planning - Hourly")

        # ===== SETS =====
        m.years = pyo.Set(initialize=range(self.start_year, self.end_year + 1))
        m.plant_types = pyo.Set(initialize=list(self.plant_params['capex'].keys()))
        m.rep_days = pyo.Set(initialize=range(self.n_rep_days))
        m.hours = pyo.Set(initialize=range(24))  # Hours 0-23
        m.start_year = self.start_year

        # ===== PARAMETERS =====
        # Plant parameters
        m.capex = pyo.Param(m.plant_types, initialize=self.plant_params['capex'])
        m.opex = pyo.Param(m.plant_types, initialize=self.plant_params['opex'])
        m.maintenance = pyo.Param(m.plant_types, initialize=self.plant_params['maintenance'])
        m.emissions = pyo.Param(m.plant_types, initialize=self.plant_params['emissions'])
        m.capacity_factor = pyo.Param(m.plant_types, initialize=self.plant_params['capacity_factor'])
        m.ramp_rate = pyo.Param(m.plant_types, initialize=self.plant_params['ramp_rate'])  # MW/min per MW

        # Lead times
        if self.use_lead_times:
            m.lead_time = pyo.Param(m.plant_types, initialize=self.plant_params['lead_time'])

        # Retirements
        if self.use_retirements:
            retirement_dict = self.calculate_retirements()
            m.retirement = pyo.Param(m.years, m.plant_types, initialize=retirement_dict)

        # Initial capacity
        m.initial_capacity = pyo.Param(m.plant_types, initialize=self.initial_cap)

        # Hourly demand (indexed by year, rep_day, hour)
        def hourly_demand_init(m, t, d, h):
            return self.hourly_data['rep_days_by_year'][t][d]['hourly_demand'][h]

        m.hourly_demand = pyo.Param(m.years, m.rep_days, m.hours, initialize=hourly_demand_init)

        # Representative day weights (constant across years)
        def rep_day_weight_init(m, d):
            first_year = self.start_year
            return self.hourly_data['rep_days_by_year'][first_year][d]['weight']

        m.rep_day_weight = pyo.Param(m.rep_days, initialize=rep_day_weight_init)

        # Peak demand (annual, for reserve margin)
        peak_demand_dict = dict(zip(self.demand_data['year'], self.demand_data['peak_demand']))
        m.peak_demand = pyo.Param(m.years, initialize=peak_demand_dict)

        # System parameters
        m.reserve_margin = pyo.Param(initialize=0.15)
        m.ramp_penalty_param = pyo.Param(initialize=self.ramp_penalty)

        # Discount factors
        discount_factors = {
            i: calculate_discount_factor(i, self.discount_rate)
            for i in range(len(m.years))
        }
        m.discount_factor = pyo.Param(range(len(m.years)), initialize=discount_factors)

        # ===== DECISION VARIABLES =====
        print("\n[Creating Decision Variables]")

        # ANNUAL VARIABLES (same as annual model)
        m.x = pyo.Var(
            m.years, m.plant_types,
            domain=pyo.NonNegativeReals,
            doc="New capacity to build (MW)"
        )

        m.N = pyo.Var(
            m.years, m.plant_types,
            domain=pyo.NonNegativeReals,
            doc="Total operating capacity (MW)"
        )

        # HOURLY VARIABLES (new for hourly model)
        m.p_hourly = pyo.Var(
            m.years, m.rep_days, m.hours, m.plant_types,
            domain=pyo.NonNegativeReals,
            doc="Hourly power output (MW)"
        )

        # Ramp violation slack variables (for soft constraints)
        if self.use_soft_ramp_constraints:
            m.ramp_violation = pyo.Var(
                m.years, m.rep_days, m.hours, m.plant_types,
                domain=pyo.NonNegativeReals,
                doc="Ramp rate constraint violation (MW)"
            )

        n_annual_vars = len(m.x) + len(m.N)
        n_hourly_vars = len(m.p_hourly)
        n_ramp_vars = len(m.ramp_violation) if self.use_soft_ramp_constraints else 0
        total_vars = n_annual_vars + n_hourly_vars + n_ramp_vars

        print(f"✓ Annual variables (x, N): {n_annual_vars:,}")
        print(f"✓ Hourly variables (p_hourly): {n_hourly_vars:,}")
        if self.use_soft_ramp_constraints:
            print(f"✓ Ramp slack variables: {n_ramp_vars:,}")
        print(f"✓ Total variables: {total_vars:,}")

        # ===== CONSTRAINTS =====
        print("\n[Adding Constraints]")

        # Add hourly constraints (demand, capacity, ramp rates)
        add_hourly_constraints(m, use_soft_ramp=self.use_soft_ramp_constraints)

        # Add annual constraints (reserve margin, capacity dynamics)
        reserve_margin_constraint(m)
        capacity_dynamics_constraint(m)

        # Add nuclear policy constraints (if enabled)
        if self.use_nuclear_policy:
            add_nuclear_policy_constraints(m, min_nuclear_share=self.min_nuclear_share)

        # ===== OBJECTIVE FUNCTION =====
        print("\n[Setting Objective Function]")

        if objective == 'cost':
            m.obj = pyo.Objective(
                expr=total_cost_objective_hourly(m),
                sense=pyo.minimize
            )
            print("✓ Objective: Minimize total cost (with ramp penalties)")

        elif objective == 'emissions':
            m.obj = pyo.Objective(
                expr=total_emissions_objective_hourly(m),
                sense=pyo.minimize
            )
            print("✓ Objective: Minimize total emissions")

        elif objective == 'multi':
            # Multi-objective requires normalization factors
            # For now, use placeholder values
            m.cost_normalization = 1e11
            m.emissions_normalization = 1e8
            cost_expr = total_cost_objective_hourly(m) / m.cost_normalization
            emissions_expr = total_emissions_objective_hourly(m) / m.emissions_normalization
            m.obj = pyo.Objective(
                expr=alpha * cost_expr + (1 - alpha) * emissions_expr,
                sense=pyo.minimize
            )
            print(f"✓ Objective: Multi-objective (α={alpha})")

        else:
            raise ValueError(f"Unknown objective: {objective}")

        self.model = m

        # Print model statistics
        n_constraints = sum(1 for _ in m.component_data_objects(pyo.Constraint, active=True))
        print(f"\n[Model Statistics]")
        print(f"✓ Variables: {total_vars:,}")
        print(f"✓ Constraints: {n_constraints:,}")
        print(f"✓ Representative days: {self.n_rep_days}")
        print(f"✓ Hours per day: 24")
        print(f"✓ Model built successfully!")

        return m

    def optimize(self, objective: str = 'cost', solver: str = 'highs',
                 time_limit: int = 600, mip_gap: float = 0.01, verbose: bool = True):
        """
        Run the hourly optimization.

        Args:
            objective: 'cost', 'emissions', or 'multi'
            solver: 'highs', 'gurobi', 'cplex'
            time_limit: Time limit in seconds (default 600 = 10 min)
            mip_gap: MIP relative gap tolerance
            verbose: Print solver output

        Returns:
            Dictionary with optimization results
        """
        # Build model if not already built
        if self.model is None:
            self.build_model(objective=objective)

        print(f"\n[Solving Hourly Model]")
        print(f"Solver: {solver}")
        print(f"Time limit: {time_limit}s")

        # Select and configure solver
        if solver == 'highs':
            opt = pyo.SolverFactory('appsi_highs')
            opt.options['time_limit'] = time_limit
            opt.options['mip_rel_gap'] = mip_gap
        elif solver == 'gurobi':
            opt = pyo.SolverFactory('gurobi')
            opt.options['TimeLimit'] = time_limit
            opt.options['MIPGap'] = mip_gap
        elif solver == 'cplex':
            opt = pyo.SolverFactory('cplex')
            opt.options['timelimit'] = time_limit
            opt.options['mipgap'] = mip_gap
        else:
            raise ValueError(f"Unknown solver: {solver}")

        # Solve
        import time
        start_time = time.time()
        print(f"Solving... (this may take several minutes)")

        self.solver_results = opt.solve(self.model, tee=verbose)
        solve_time = time.time() - start_time

        # Check solution status
        termination = self.solver_results.solver.termination_condition

        if termination == pyo.TerminationCondition.optimal:
            print(f"\n✓ Optimal solution found!")
            print(f"  Solve time: {solve_time:.1f} seconds")
        elif termination == pyo.TerminationCondition.feasible:
            print(f"\n✓ Feasible solution found (not proven optimal)")
            print(f"  Solve time: {solve_time:.1f} seconds")
        else:
            print(f"\n✗ Solver failed: {termination}")
            return None

        # Extract results
        self.results = self.get_results()
        return self.results

    def get_results(self) -> Dict:
        """
        Extract and format optimization results.

        Returns:
            Dictionary with capacity, generation, hourly dispatch, costs, emissions
        """
        if self.model is None:
            raise ValueError("No model to extract results from")

        m = self.model
        print("\n[Extracting Results]")

        # Extract annual variables (same as annual model)
        new_builds = []
        capacity = []

        for t in m.years:
            for i in m.plant_types:
                # Handle uninitialized variables (set to 0)
                try:
                    x_val = pyo.value(m.x[t, i])
                    if x_val is None:
                        x_val = 0.0
                except:
                    x_val = 0.0

                try:
                    N_val = pyo.value(m.N[t, i])
                    if N_val is None:
                        N_val = 0.0
                except:
                    N_val = 0.0

                new_builds.append({
                    'year': t,
                    'plant_type': i,
                    'new_capacity_MW': x_val
                })

                capacity.append({
                    'year': t,
                    'plant_type': i,
                    'total_capacity_MW': N_val
                })

        # Extract hourly generation
        hourly_generation = []
        for t in m.years:
            for d in m.rep_days:
                for h in m.hours:
                    for i in m.plant_types:
                        try:
                            p_val = pyo.value(m.p_hourly[t, d, h, i])
                            if p_val is None:
                                p_val = 0.0
                        except:
                            p_val = 0.0
                        hourly_generation.append({
                            'year': t,
                            'rep_day': d,
                            'hour': h,
                            'plant_type': i,
                            'generation_MW': p_val
                        })

        # Convert to DataFrames
        df_builds = pd.DataFrame(new_builds)
        df_capacity = pd.DataFrame(capacity)
        df_hourly = pd.DataFrame(hourly_generation)

        # Calculate annual generation from hourly (weighted by rep day weights)
        annual_generation = []
        for t in m.years:
            for i in m.plant_types:
                # Sum over all rep days and hours, weighted
                gen_mwh = 0.0
                for d in m.rep_days:
                    weight = pyo.value(m.rep_day_weight[d])
                    for h in m.hours:
                        try:
                            p_val = pyo.value(m.p_hourly[t, d, h, i])
                            if p_val is not None:
                                gen_mwh += p_val * weight
                        except:
                            pass  # Uninitialized variable, skip
                annual_generation.append({
                    'year': t,
                    'plant_type': i,
                    'generation_MWh': gen_mwh
                })

        df_generation = pd.DataFrame(annual_generation)

        # Calculate totals
        print("✓ Calculating costs and emissions...")

        # Summary statistics
        total_new_capacity = df_builds['new_capacity_MW'].sum()
        final_capacity = df_capacity[df_capacity['year'] == self.end_year]['total_capacity_MW'].sum()
        total_generation = df_generation['generation_MWh'].sum()

        # Calculate individual cost components first
        capex_cost, opex_cost, maintenance_cost, ramp_penalty_cost = self._calculate_cost_components(
            df_builds, df_capacity, df_generation, m
        )

        # Total cost is always the sum of components (don't use objective value
        # because objective might be emissions or multi-objective)
        total_cost = capex_cost + opex_cost + maintenance_cost + ramp_penalty_cost

        # Calculate emissions
        total_emissions = self._calculate_emissions_from_data(df_generation)

        # Check for ramp violations
        if self.use_soft_ramp_constraints:
            total_ramp_violations = 0.0
            for t in m.years:
                for d in m.rep_days:
                    for h in m.hours:
                        for i in m.plant_types:
                            # Skip hour 0 - no ramp constraints there
                            if h == 0:
                                continue
                            try:
                                val = pyo.value(m.ramp_violation[t, d, h, i])
                                if val is not None:
                                    total_ramp_violations += val
                            except:
                                # Uninitialized variable - no violation
                                pass
            print(f"  Ramp violations: {total_ramp_violations:.0f} MW total")
        else:
            total_ramp_violations = 0.0

        summary = {
            'total_cost_npv': total_cost,
            'total_cost_billions': total_cost / 1e9,
            'total_emissions_tons': total_emissions,
            'total_emissions_megatons': total_emissions / 1e6,
            'total_new_capacity_MW': total_new_capacity,
            'total_new_capacity_GW': total_new_capacity / 1000,
            'final_capacity_MW': final_capacity,
            'final_capacity_GW': final_capacity / 1000,
            'total_generation_MWh': total_generation,
            'total_generation_TWh': total_generation / 1e6,
            'planning_horizon_years': self.end_year - self.start_year + 1,
            'total_ramp_violations_MW': total_ramp_violations,
        }

        print(f"\n[Results Summary]")
        print(f"Total Cost (NPV): ${summary['total_cost_billions']:.2f} billion")
        print(f"Total Emissions: {summary['total_emissions_megatons']:.2f} megatons CO2")
        print(f"New Capacity: {summary['total_new_capacity_GW']:.2f} GW")
        print(f"Final Capacity: {summary['final_capacity_GW']:.2f} GW")
        print(f"Total Generation: {summary['total_generation_TWh']:.2f} TWh")

        return {
            'total_cost': total_cost,
            'total_emissions': total_emissions,
            'capex_cost': capex_cost,
            'opex_cost': opex_cost,
            'maintenance_cost': maintenance_cost,
            'ramp_penalty_cost': ramp_penalty_cost,
            'new_builds': df_builds,
            'capacity': self._pivot_capacity(df_capacity),
            'generation': df_generation,
            'hourly_generation': df_hourly,
            'summary': summary
        }

    def _pivot_capacity(self, df_capacity):
        """Convert capacity DataFrame to dict of {year: {plant_type: capacity}}."""
        capacity_dict = {}
        for year in df_capacity['year'].unique():
            year_data = df_capacity[df_capacity['year'] == year]
            capacity_dict[year] = {
                row['plant_type']: row['total_capacity_MW']
                for _, row in year_data.iterrows()
            }
        return capacity_dict

    def _calculate_cost_components(self, df_builds, df_capacity, df_generation, model):
        """Calculate individual cost components (capex, opex, maintenance, ramp penalty)."""
        capex_cost = 0.0
        opex_cost = 0.0
        maintenance_cost = 0.0
        ramp_penalty_cost = 0.0

        for year_idx, year in enumerate(range(self.start_year, self.end_year + 1)):
            discount_factor = calculate_discount_factor(year_idx, self.discount_rate)

            # Capital costs
            year_builds = df_builds[df_builds['year'] == year]
            for _, row in year_builds.iterrows():
                plant_type = row['plant_type']
                new_cap_mw = row['new_capacity_MW']
                capex_per_kw = self.plant_params['capex'][plant_type]
                capex_cost += new_cap_mw * capex_per_kw * 1000 * discount_factor

            # Operating costs
            year_gen = df_generation[df_generation['year'] == year]
            for _, row in year_gen.iterrows():
                plant_type = row['plant_type']
                gen_mwh = row['generation_MWh']
                opex_per_mwh = self.plant_params['opex'][plant_type]
                opex_cost += gen_mwh * opex_per_mwh * discount_factor

            # Maintenance costs
            year_cap = df_capacity[df_capacity['year'] == year]
            for _, row in year_cap.iterrows():
                plant_type = row['plant_type']
                cap_mw = row['total_capacity_MW']
                maint_per_mw_year = self.plant_params['maintenance'][plant_type]
                maintenance_cost += cap_mw * maint_per_mw_year * discount_factor

        # Ramp penalty cost
        if self.use_soft_ramp_constraints and hasattr(model, 'ramp_violation'):
            for year_idx, year in enumerate(range(self.start_year, self.end_year + 1)):
                discount_factor = calculate_discount_factor(year_idx, self.discount_rate)
                for d in model.rep_days:
                    weight = pyo.value(model.rep_day_weight[d])
                    for h in model.hours:
                        if h == 0:  # Skip hour 0
                            continue
                        for i in model.plant_types:
                            try:
                                violation = pyo.value(model.ramp_violation[year, d, h, i])
                                if violation is not None and violation > 0:
                                    ramp_penalty_cost += violation * self.ramp_penalty * weight * discount_factor
                            except:
                                pass

        return capex_cost, opex_cost, maintenance_cost, ramp_penalty_cost

    def _calculate_cost_from_data(self, df_builds, df_capacity, df_generation):
        """Calculate total NPV cost from extracted data."""
        total_cost = 0.0

        for year_idx, year in enumerate(range(self.start_year, self.end_year + 1)):
            discount_factor = calculate_discount_factor(year_idx, self.discount_rate)

            # Capital costs
            year_builds = df_builds[df_builds['year'] == year]
            for _, row in year_builds.iterrows():
                plant_type = row['plant_type']
                new_cap_mw = row['new_capacity_MW']
                capex_per_kw = self.plant_params['capex'][plant_type]
                total_cost += new_cap_mw * capex_per_kw * 1000 * discount_factor

            # Operating costs
            year_gen = df_generation[df_generation['year'] == year]
            for _, row in year_gen.iterrows():
                plant_type = row['plant_type']
                gen_mwh = row['generation_MWh']
                opex_per_mwh = self.plant_params['opex'][plant_type]
                total_cost += gen_mwh * opex_per_mwh * discount_factor

            # Maintenance costs
            year_cap = df_capacity[df_capacity['year'] == year]
            for _, row in year_cap.iterrows():
                plant_type = row['plant_type']
                cap_mw = row['total_capacity_MW']
                maint_per_mw_year = self.plant_params['maintenance'][plant_type]
                total_cost += cap_mw * maint_per_mw_year * discount_factor

        return total_cost

    def _calculate_emissions_from_data(self, df_generation):
        """Calculate total emissions from extracted data."""
        total_emissions = 0.0
        for _, row in df_generation.iterrows():
            plant_type = row['plant_type']
            gen_mwh = row['generation_MWh']
            emission_factor = self.plant_params['emissions'][plant_type]
            total_emissions += gen_mwh * emission_factor
        return total_emissions

    def solve(self, solver: str = 'highs', time_limit: int = 600,
              mip_gap: float = 0.01, tee: bool = False, save_results: bool = True):
        """
        Solve the optimization model.

        Args:
            solver: Solver to use ('highs', 'gurobi', 'cplex')
            time_limit: Time limit in seconds (default 600 = 10 min)
            mip_gap: MIP relative gap tolerance (default 1%)
            tee: Print solver output (default False)
            save_results: Save results to self.results (default True)

        Returns:
            Dictionary with:
            - status: 'optimal', 'feasible', or 'failed'
            - solve_time: Time to solve (seconds)
            - objective_value: Objective function value
            - termination_condition: Pyomo termination condition
        """
        if self.model is None:
            raise ValueError("No model to solve. Run build_model() first.")

        print(f"\n[Solving Hourly Model]")
        print(f"Solver: {solver}")
        print(f"Time limit: {time_limit}s")

        # Select and configure solver
        if solver == 'highs':
            opt = pyo.SolverFactory('appsi_highs')
            opt.options['time_limit'] = time_limit
            opt.options['mip_rel_gap'] = mip_gap
        elif solver == 'gurobi':
            opt = pyo.SolverFactory('gurobi')
            opt.options['TimeLimit'] = time_limit
            opt.options['MIPGap'] = mip_gap
        else:
            raise ValueError(f"Unknown solver: {solver}. Use 'highs' or 'gurobi'.")

        # Solve
        import time
        start_time = time.time()
        print(f"Solving...")

        self.solver_results = opt.solve(self.model, tee=tee)
        solve_time = time.time() - start_time

        # Check solution status
        termination = self.solver_results.solver.termination_condition

        if termination == pyo.TerminationCondition.optimal:
            status = 'optimal'
            print(f"\n✓ Optimal solution found in {solve_time:.1f}s")
        elif termination == pyo.TerminationCondition.feasible:
            status = 'feasible'
            print(f"\n✓ Feasible solution found in {solve_time:.1f}s (not proven optimal)")
        else:
            status = 'failed'
            print(f"\n✗ Solver failed: {termination}")
            return {
                'status': status,
                'solve_time': solve_time,
                'objective_value': None,
                'termination_condition': str(termination)
            }

        # Get objective value
        try:
            obj_value = pyo.value(self.model.obj)
        except:
            obj_value = None
            print("Warning: Could not extract objective value")

        # Extract and save results if requested
        if save_results:
            self.results = self.get_results()
        else:
            self.results = None

        return {
            'status': status,
            'solve_time': solve_time,
            'objective_value': obj_value,
            'termination_condition': str(termination)
        }
