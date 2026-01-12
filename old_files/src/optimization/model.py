"""
Main optimization model for power system planning.

This module contains the PowerSystemOptimization class which coordinates
the entire optimization process.
"""

import pyomo.environ as pyo
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path

from .variables import define_variables
from .objectives import total_cost_objective, total_emissions_objective, multi_objective
from .constraints import add_all_constraints
from ..utils.financial import calculate_discount_factor


class PowerSystemOptimization:
    """
    Main optimization model for Ontario power system planning.

    Determines optimal mix of power plants to build over 20 years (2025-2045)
    to minimize costs and emissions while meeting demand constraints.
    """

    def __init__(self, start_year: int = 2025, end_year: int = 2045,
                 data_path: str = 'data/processed/',
                 discount_rate: float = 0.0392,
                 use_lead_times: bool = True,
                 use_retirements: bool = True):
        """
        Initialize the optimization model.

        Args:
            start_year: First year of planning horizon
            end_year: Last year of planning horizon
            data_path: Path to processed data files
            discount_rate: Real discount rate (default 3.92%)
            use_lead_times: Include construction lead times (default True)
            use_retirements: Include plant retirements (default True)
        """
        self.start_year = start_year
        self.end_year = end_year
        self.data_path = Path(data_path)
        self.discount_rate = discount_rate
        self.use_lead_times = use_lead_times
        self.use_retirements = use_retirements
        self.model = None
        self.results = None
        self.solver_results = None

        # Data containers
        self.plant_params = None
        self.demand_data = None
        self.initial_cap = None

    def load_data(self):
        """
        Load all required data from files.

        Loads:
        - Plant parameters (costs, emissions, capacity factors, etc.)
        - Demand forecast
        - Initial capacity
        """
        import json

        # Load plant parameters
        with open(self.data_path / 'plant_parameters.json', 'r') as f:
            self.plant_params = json.load(f)

        # Load demand forecast
        self.demand_data = pd.read_csv(self.data_path / 'demand_forecast.csv')

        # Filter demand data to planning horizon
        self.demand_data = self.demand_data[
            (self.demand_data['year'] >= self.start_year) &
            (self.demand_data['year'] <= self.end_year)
        ].reset_index(drop=True)

        # Load initial capacity
        with open(self.data_path / 'initial_capacity.json', 'r') as f:
            self.initial_cap = json.load(f)

        print(f"✓ Loaded data for {len(self.demand_data)} years ({self.start_year}-{self.end_year})")
        print(f"✓ Plant types: {list(self.plant_params['capex'].keys())}")

    def calculate_retirements(self):
        """
        Calculate plant retirements based on initial capacity age and lifespan.

        Assumes initial capacity was built uniformly over its lifespan.
        For simplicity, retires 1/lifespan of initial capacity each year.

        Returns:
            Dictionary {(year, plant_type): retirement_MW}
        """
        retirements = {}

        for year in range(self.start_year, self.end_year + 1):
            for plant_type in self.plant_params['capex'].keys():
                lifespan = self.plant_params['lifespan'][plant_type]
                initial_cap = self.initial_cap[plant_type]

                # Simple retirement model: retire 1/lifespan of initial capacity per year
                # This assumes initial plants were built uniformly over past lifespan years
                annual_retirement = initial_cap / lifespan

                # Only retire if we're far enough into planning horizon
                # and if there's still initial capacity left
                years_elapsed = year - self.start_year
                if years_elapsed < lifespan:
                    retirements[(year, plant_type)] = annual_retirement
                else:
                    # All initial capacity has been retired
                    retirements[(year, plant_type)] = 0.0

        return retirements

    def build_model(self, objective: str = 'cost', alpha: float = 0.5):
        """
        Build the Pyomo optimization model.

        Args:
            objective: Objective function to use ('cost', 'emissions', or 'multi')
            alpha: Weight for multi-objective (only used if objective='multi')
        """
        if self.plant_params is None:
            self.load_data()

        print(f"\n[Building Model]")
        print(f"Objective: {objective}")
        if self.use_lead_times:
            print(f"✓ Including construction lead times")
        if self.use_retirements:
            print(f"✓ Including plant retirements")

        # Create concrete model
        m = pyo.ConcreteModel(name="Ontario Power Planning")

        # ===== SETS =====
        m.years = pyo.Set(initialize=range(self.start_year, self.end_year + 1))
        m.plant_types = pyo.Set(initialize=list(self.plant_params['capex'].keys()))
        m.start_year = self.start_year

        # ===== PARAMETERS =====
        # Plant parameters
        m.capex = pyo.Param(m.plant_types, initialize=self.plant_params['capex'])
        m.opex = pyo.Param(m.plant_types, initialize=self.plant_params['opex'])
        m.maintenance = pyo.Param(m.plant_types, initialize=self.plant_params['maintenance'])
        m.emissions = pyo.Param(m.plant_types, initialize=self.plant_params['emissions'])
        m.capacity_factor = pyo.Param(m.plant_types, initialize=self.plant_params['capacity_factor'])

        # Add lead times if enabled
        if self.use_lead_times:
            m.lead_time = pyo.Param(m.plant_types, initialize=self.plant_params['lead_time'])

        # Add retirements if enabled
        if self.use_retirements:
            retirement_dict = self.calculate_retirements()
            m.retirement = pyo.Param(m.years, m.plant_types, initialize=retirement_dict)

        # Initial capacity
        m.initial_capacity = pyo.Param(m.plant_types, initialize=self.initial_cap)

        # Demand parameters (convert to dictionary indexed by year)
        annual_demand_dict = dict(zip(self.demand_data['year'],
                                     self.demand_data['annual_demand'] * 1000))  # GWh to MWh
        peak_demand_dict = dict(zip(self.demand_data['year'],
                                   self.demand_data['peak_demand']))  # MW

        m.annual_demand = pyo.Param(m.years, initialize=annual_demand_dict)
        m.peak_demand = pyo.Param(m.years, initialize=peak_demand_dict)

        # System parameters
        m.reserve_margin = pyo.Param(initialize=0.15)  # 15%

        # Discount factors
        discount_factors = {
            i: calculate_discount_factor(i, self.discount_rate)
            for i in range(len(m.years))
        }
        m.discount_factor = pyo.Param(range(len(m.years)), initialize=discount_factors)

        # ===== DECISION VARIABLES =====
        define_variables(m)

        # ===== CONSTRAINTS =====
        add_all_constraints(m)

        # ===== OBJECTIVE FUNCTION =====
        if objective == 'cost':
            m.obj = pyo.Objective(expr=total_cost_objective(m), sense=pyo.minimize)
            print("✓ Using cost minimization objective")
        elif objective == 'emissions':
            m.obj = pyo.Objective(expr=total_emissions_objective(m), sense=pyo.minimize)
            print("✓ Using emissions minimization objective")
        elif objective == 'multi':
            # Need to run single objectives first to get normalization factors
            print("✓ Using multi-objective (weighted sum)")
            # For now, use placeholder normalization
            m.cost_normalization = 1e11  # ~$100B
            m.emissions_normalization = 1e8  # ~100M tons
            m.obj = pyo.Objective(expr=multi_objective(m, alpha), sense=pyo.minimize)
        else:
            raise ValueError(f"Unknown objective: {objective}")

        self.model = m
        print(f"✓ Model built successfully")
        print(f"  - Variables: {len(m.x) + len(m.p) + len(m.N)}")
        print(f"  - Constraints: {sum(1 for _ in m.component_data_objects(pyo.Constraint))}")

        return m

    def optimize(self, objective: str = 'cost', solver: str = 'highs',
                 time_limit: int = 300, mip_gap: float = 0.01):
        """
        Run the optimization.

        Args:
            objective: Objective function to use ('cost', 'emissions', 'multi')
            solver: Solver to use ('highs', 'gurobi', 'cplex')
            time_limit: Time limit in seconds (default 300)
            mip_gap: MIP relative gap tolerance (default 1%)

        Returns:
            Dictionary with optimization results
        """
        # Build model if not already built
        if self.model is None:
            self.build_model(objective=objective)

        print(f"\n[Solving Model]")
        print(f"Solver: {solver}")

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
        print(f"Solving... (this may take a few minutes)")
        self.solver_results = opt.solve(self.model, tee=True)

        # Check solution status
        if self.solver_results.solver.termination_condition == pyo.TerminationCondition.optimal:
            print(f"\n✓ Optimal solution found!")
        elif self.solver_results.solver.termination_condition == pyo.TerminationCondition.feasible:
            print(f"\n✓ Feasible solution found (not proven optimal)")
        else:
            print(f"\n✗ Solver failed: {self.solver_results.solver.termination_condition}")
            return None

        # Extract results
        self.results = self.get_results()
        return self.results

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

    def get_results(self) -> Dict:
        """
        Extract and format optimization results.

        Returns:
            Dictionary containing:
            - total_cost: Total NPV cost ($)
            - total_emissions: Total emissions (tons CO2)
            - new_builds: DataFrame of new capacity additions
            - capacity: DataFrame of total capacity by year and type
            - generation: DataFrame of generation by year and type
            - summary: Dictionary with key metrics
        """
        if self.model is None:
            raise ValueError("No model to extract results from. Run optimize() first.")

        m = self.model

        # Extract decision variable values
        new_builds = []
        capacity = []
        generation = []

        for t in m.years:
            for i in m.plant_types:
                # New builds - handle uninitialized variables (fixed by solver at 0)
                try:
                    x_val = pyo.value(m.x[t, i])
                    if x_val is None:
                        x_val = 0.0
                except:
                    x_val = 0.0

                new_builds.append({
                    'year': t,
                    'plant_type': i,
                    'new_capacity_MW': x_val
                })

                # Total capacity
                try:
                    N_val = pyo.value(m.N[t, i])
                    if N_val is None:
                        N_val = 0.0
                except:
                    N_val = 0.0

                capacity.append({
                    'year': t,
                    'plant_type': i,
                    'total_capacity_MW': N_val
                })

                # Generation
                try:
                    p_val = pyo.value(m.p[t, i])
                    if p_val is None:
                        p_val = 0.0
                except:
                    p_val = 0.0

                generation.append({
                    'year': t,
                    'plant_type': i,
                    'generation_MWh': p_val
                })

        # Convert to DataFrames
        df_builds = pd.DataFrame(new_builds)
        df_capacity = pd.DataFrame(capacity)
        df_generation = pd.DataFrame(generation)

        # Calculate objective values manually from the extracted data
        # (safer than trying to evaluate Pyomo expressions with fixed variables)
        total_cost = self._calculate_cost_from_data(df_builds, df_capacity, df_generation)
        total_emissions = self._calculate_emissions_from_data(df_generation)

        # Calculate summary statistics
        total_new_capacity = df_builds['new_capacity_MW'].sum()
        final_capacity = df_capacity[df_capacity['year'] == self.end_year]['total_capacity_MW'].sum()
        total_generation = df_generation['generation_MWh'].sum()

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
        }

        # Print summary
        print(f"\n[Results Summary]")
        print(f"Total Cost (NPV): ${summary['total_cost_billions']:.2f} billion")
        print(f"Total Emissions: {summary['total_emissions_megatons']:.2f} megatons CO2")
        print(f"New Capacity Built: {summary['total_new_capacity_GW']:.2f} GW")
        print(f"Final Total Capacity: {summary['final_capacity_GW']:.2f} GW")
        print(f"Total Generation: {summary['total_generation_TWh']:.2f} TWh")

        return {
            'total_cost': total_cost,
            'total_emissions': total_emissions,
            'new_builds': df_builds,
            'capacity': df_capacity,
            'generation': df_generation,
            'summary': summary
        }
