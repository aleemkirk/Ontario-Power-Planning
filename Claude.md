# Ontario Power Plant Optimization Project

## Project Overview

Build a multi-objective optimization model to determine the optimal mix of power plants Ontario should construct over the next 20 years (2025-2045) to meet growing electricity demand while minimizing costs and carbon emissions.

## Problem Statement

Ontario's electricity demand is projected to grow 75% by 2050 (from 151 TWh to 263 TWh annually). We need to optimize:

**Primary Objectives:**
1. **Minimize Total System Cost** (capital + operating + maintenance)
2. **Minimize Carbon Emissions**

**Subject to Constraints:**
- Meet hourly electricity demand
- Maintain 15% reserve margin above peak demand
- Respect ramp rate limits (ability to respond to demand changes)
- Account for plant construction lead times and lifespans

**Plant Types:** Nuclear, Wind, Solar, Natural Gas, Hydro, Biofuel

**Approach:** Generate a Pareto frontier showing cost-emissions trade-offs to inform policy decisions.

## Technical Architecture

### Core Components

1. **Data Layer** (`data/`)
   - Load Ontario energy system data (capacities, costs, demand forecasts)
   - Define plant parameters (capital costs, operating costs, capacity factors, ramp rates, emissions)
   - Manage time series data (demand profiles, weather patterns if needed)

2. **Optimization Model** (`src/optimization/`)
   - Decision variables: Number of plants to build each year, hourly generation levels
   - Objective functions: Total cost (NPV), total emissions
   - Constraints: Demand satisfaction, capacity limits, ramp rates, reserve margin
   - Multi-objective solver to generate Pareto frontier

3. **Solver Interface** (`src/solvers/`)
   - Support for multiple solvers: HiGHS (open-source), Gurobi, CPLEX
   - Abstract interface to switch between solvers easily

4. **Analysis & Visualization** (`src/analysis/`)
   - Pareto frontier visualization
   - Capacity expansion timelines
   - Cost breakdowns by component
   - Emissions trajectories
   - Sensitivity analysis tools

5. **Results & Reporting** (`results/`)
   - Export optimization results to CSV/JSON
   - Generate summary reports
   - Create interactive dashboards (optional: Plotly Dash or Streamlit)

## Project Structure

```
ontario-power-optimization/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── claude.md                    # This file
├── data/
│   ├── raw/
│   │   └── Ontario_Energy_Data_Summary.md  # Reference data
│   ├── processed/
│   │   ├── plant_parameters.json
│   │   ├── demand_forecast.csv
│   │   └── initial_capacity.json
│   └── load_data.py
├── src/
│   ├── __init__.py
│   ├── optimization/
│   │   ├── __init__.py
│   │   ├── model.py              # Main optimization model
│   │   ├── objectives.py         # Cost and emissions objectives
│   │   ├── constraints.py        # All constraint definitions
│   │   └── variables.py          # Decision variable definitions
│   ├── solvers/
│   │   ├── __init__.py
│   │   ├── base_solver.py        # Abstract solver interface
│   │   ├── highs_solver.py       # HiGHS solver
│   │   └── gurobi_solver.py      # Gurobi solver (optional)
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── pareto.py             # Pareto frontier generation
│   │   ├── visualizations.py     # Plotting functions
│   │   └── sensitivity.py        # Sensitivity analysis
│   └── utils/
│       ├── __init__.py
│       ├── financial.py          # NPV calculations, discount rates
│       └── time_series.py        # Time series utilities
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_prototype_model.ipynb
│   ├── 03_full_model.ipynb
│   └── 04_results_analysis.ipynb
├── tests/
│   ├── __init__.py
│   ├── test_model.py
│   ├── test_constraints.py
│   └── test_objectives.py
├── results/
│   ├── figures/
│   ├── data/
│   └── reports/
└── scripts/
    ├── run_optimization.py       # Main execution script
    ├── generate_pareto.py        # Pareto frontier generation
    └── create_report.py          # Report generation
```

## Implementation Phases

### Phase 1: Setup & Data Preparation (Day 1)

**Goal:** Set up project structure and load all data

**Tasks:**
1. Create project directory structure
2. Set up Python environment with dependencies
3. Create `data/processed/plant_parameters.json` with all plant data:
   - Capital costs, operating costs, maintenance costs
   - Capacity factors, ramp rates, emission factors
   - Construction lead times, lifespans
4. Create `data/processed/demand_forecast.csv` with demand projections (2025-2045)
5. Create `data/processed/initial_capacity.json` with 2025 starting capacities
6. Write `data/load_data.py` to load and validate all data

**Key Files to Create:**
- `requirements.txt` - Python dependencies
- `data/load_data.py` - Data loading utilities
- `data/processed/*.json` - All parameter files

### Phase 2: Prototype Model (Day 2-3)

**Goal:** Build simplified 5-year optimization model to validate approach

**Simplifications for Prototype:**
- 5-year horizon (2025-2030) instead of 20 years
- Annual time resolution instead of hourly
- Single objective first (minimize cost), add emissions later
- Ignore ramp rate constraints initially
- Use smaller solver (HiGHS)

**Tasks:**
1. Implement decision variables in `src/optimization/variables.py`
2. Implement cost objective in `src/optimization/objectives.py`
3. Implement basic constraints in `src/optimization/constraints.py`:
   - Demand satisfaction
   - Capacity constraints
   - Reserve margin
4. Create main model in `src/optimization/model.py`
5. Test with simple solver in `src/solvers/highs_solver.py`
6. Validate results make sense

**Expected Output:**
- Working optimization model that suggests plant builds
- Total cost calculation
- Basic constraint satisfaction

### Phase 3: Full Single-Objective Model (Day 4-5)

**Goal:** Extend to 20 years with all constraints

**Enhancements:**
- Extend time horizon to 20 years (2025-2045)
- Add construction lead times
- Add ramp rate constraints
- Add plant retirement logic
- Refine to monthly or representative day resolution (optional)

**Tasks:**
1. Update model to handle 20-year horizon
2. Implement lead time constraints (can't use plants until after construction)
3. Implement ramp rate constraints in `src/optimization/constraints.py`
4. Add plant lifespan and retirement logic
5. Improve solver performance (warm starts, variable bounds)
6. Run and validate 20-year optimization

**Expected Output:**
- 20-year capacity expansion plan
- Year-by-year build schedule
- Total system cost breakdown
- Validation that all constraints are satisfied

### Phase 4: Multi-Objective Optimization (Day 6-7)

**Goal:** Add emissions objective and generate Pareto frontier

**Tasks:**
1. Implement emissions objective in `src/optimization/objectives.py`
2. Implement weighted sum approach for multi-objective optimization
3. Create Pareto frontier generation in `src/analysis/pareto.py`:
   - Vary weight α from 0 to 1 (cost vs emissions)
   - Run optimization for each weight value
   - Store all Pareto-optimal solutions
4. Implement ε-constraint method as alternative
5. Normalize objectives for fair weighting

**Expected Output:**
- Set of Pareto-optimal solutions
- Each solution shows: total cost, total emissions, capacity mix
- Trade-off analysis: cost of reducing emissions by X tons

### Phase 5: Analysis & Visualization (Day 8-9)

**Goal:** Create comprehensive visualizations and reports

**Tasks:**
1. Implement visualization functions in `src/analysis/visualizations.py`:
   - Pareto frontier plot (cost vs emissions)
   - Capacity expansion timeline (stacked area chart)
   - Generation mix by year (stacked bar chart)
   - Cost breakdown (capital vs operating vs maintenance)
   - Emissions trajectory over time
   - Marginal cost of carbon reduction
2. Create sensitivity analysis in `src/analysis/sensitivity.py`:
   - Impact of discount rate (3%, 4%, 5%)
   - Impact of demand growth (±20%)
   - Impact of technology costs (±30%)
3. Generate summary statistics for each Pareto solution
4. Export results to CSV/JSON

**Expected Output:**
- Publication-quality figures
- Interactive visualizations (optional)
- Sensitivity analysis results
- Exportable data files

### Phase 6: Documentation & Testing (Day 10)

**Goal:** Finalize project with documentation and tests

**Tasks:**
1. Write comprehensive README.md
2. Add docstrings to all functions
3. Write unit tests in `tests/`:
   - Test constraint satisfaction
   - Test objective calculations
   - Test data loading
4. Create example scripts in `scripts/`
5. Create Jupyter notebooks demonstrating usage
6. Add configuration file for easy parameter adjustment

**Expected Output:**
- Well-documented codebase
- Working test suite
- Example notebooks
- Easy-to-use scripts

## Key Data Parameters

All data has been collected and is available in `Ontario_Energy_Data_Summary.md`. Here are the key parameters to implement:

### Plant Parameters

```python
plant_types = ['nuclear', 'wind', 'solar', 'gas', 'hydro', 'biofuel']

# Capital costs ($/kW)
capex = {
    'nuclear': 17500,
    'wind': 1900,
    'solar': 1300,
    'gas': 1500,
    'hydro': 3000,
    'biofuel': 3200
}

# Operating costs ($/MWh)
opex = {
    'nuclear': 22,
    'wind': 12,
    'solar': 12,
    'gas': 55,
    'hydro': 10,
    'biofuel': 42
}

# Maintenance costs ($/MW/year)
maintenance = {
    'nuclear': 105000,
    'wind': 45000,
    'solar': 20000,
    'gas': 20000,
    'hydro': 37500,
    'biofuel': 70000
}

# Emission factors (tons CO2/MWh)
emissions = {
    'nuclear': 0.012,
    'wind': 0.011,
    'solar': 0.048,
    'gas': 0.45,
    'hydro': 0.024,
    'biofuel': 0.23
}

# Capacity factors
capacity_factor = {
    'nuclear': 0.90,
    'wind': 0.35,
    'solar': 0.15,
    'gas': 0.55,
    'hydro': 0.50,
    'biofuel': 0.80
}

# Ramp rates (MW/min per MW of capacity)
ramp_rate = {
    'nuclear': 0.02,
    'wind': 0.05,
    'solar': 0.10,
    'gas': 0.04,
    'hydro': 0.15,
    'biofuel': 0.01
}

# Construction lead times (years)
lead_time = {
    'nuclear': 7,
    'wind': 2,
    'solar': 2,
    'gas': 3,
    'hydro': 6,
    'biofuel': 3
}

# Plant lifespans (years)
lifespan = {
    'nuclear': 60,
    'wind': 25,
    'solar': 30,
    'gas': 35,
    'hydro': 100,
    'biofuel': 35
}
```

### System Parameters

```python
# Financial
discount_rate_real = 0.0392  # 3.92% real discount rate
inflation_rate = 0.02  # 2% annual inflation (for reference, not used in real model)

# Time horizon
start_year = 2025
end_year = 2045
years = range(start_year, end_year + 1)

# Initial capacity (MW) - 2025
initial_capacity = {
    'nuclear': 13000,
    'wind': 5575,
    'solar': 2669,
    'gas': 10500,
    'hydro': 8500,
    'biofuel': 205
}

# Demand
demand_2025 = 151000  # GWh/year (151 TWh)
demand_growth_rate = 0.022  # 2.2% annual growth
peak_demand_2025 = 24000  # MW
reserve_margin = 0.15  # 15% above peak demand
```

## Mathematical Formulation

### Decision Variables

```
x[t,i] = Number of plants of type i to build in year t (integer or continuous)
p[t,i,h] = Power output from plant type i in year t, hour h (MW)
N[t,i] = Total operating capacity of plant type i in year t (MW)
```

### Objective Functions

**1. Minimize Total Cost (Z1)**
```
Z1 = Σ_t Σ_i (x[t,i] × CapEx[i] × Capacity[i]) / (1 + r)^t
   + Σ_t Σ_i Σ_h (p[t,i,h] × OpEx[i]) / (1 + r)^t
   + Σ_t Σ_i (N[t,i] × MainEx[i]) / (1 + r)^t
```

**2. Minimize Emissions (Z2)**
```
Z2 = Σ_t Σ_i Σ_h (p[t,i,h] × EmissionFactor[i])
```

**Multi-Objective:**
```
Minimize: α × (Z1/Z1_max) + (1-α) × (Z2/Z2_max)
where α ∈ [0, 1]
```

### Constraints

**1. Demand Satisfaction**
```
Σ_i p[t,i,h] ≥ Demand[t,h]  ∀t,h
```

**2. Capacity Constraint**
```
p[t,i,h] ≤ N[t,i] × CapacityFactor[i]  ∀t,i,h
```

**3. Reserve Margin**
```
Σ_i N[t,i] ≥ (1 + ReserveMargin) × PeakDemand[t]  ∀t
```

**4. Ramp Rate Constraint**
```
|p[t,i,h] - p[t,i,h-1]| ≤ RampRate[i] × N[t,i]  ∀t,i,h
```

**5. Capacity Dynamics**
```
N[t,i] = N[t-1,i] + x[t-LeadTime[i],i] × Capacity[i] - Retirements[t,i]  ∀t,i
```

**6. Non-negativity**
```
x[t,i] ≥ 0  ∀t,i
p[t,i,h] ≥ 0  ∀t,i,h
```

## Technology Stack

### Required Python Packages

```
# Core optimization
pyomo>=6.7.0          # Optimization modeling language
highspy>=1.7.0        # HiGHS solver (open-source, fast)

# Optional commercial solvers (better performance)
gurobipy>=11.0.0      # Gurobi solver (free academic license)
# cplex>=22.1.0       # IBM CPLEX solver

# Data manipulation
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.11.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0        # Interactive plots

# Utilities
tqdm>=4.65.0          # Progress bars
pyyaml>=6.0           # Configuration files
python-dotenv>=1.0.0  # Environment variables

# Development
pytest>=7.3.0         # Testing
black>=23.3.0         # Code formatting
jupyter>=1.0.0        # Notebooks
```

### Recommended IDE Setup
- VS Code with Python extension
- Claude Code CLI for AI assistance
- Git for version control

## Solver Selection Guide

### HiGHS (Recommended for Development)
- **Pros:** Open-source, fast, no license needed, good for LP/MILP
- **Cons:** Not as fast as commercial solvers for large problems
- **Use for:** Prototype, testing, small-to-medium problems

### Gurobi (Recommended for Production)
- **Pros:** Very fast, excellent for MILP, free academic license
- **Cons:** Requires license for commercial use
- **Use for:** Large-scale problems, final optimization runs

### CPLEX
- **Pros:** Industry standard, very robust
- **Cons:** Expensive, complex licensing
- **Use for:** Enterprise applications

## Expected Outputs

### 1. Optimization Results
- **Pareto frontier:** 10-20 optimal solutions showing cost-emissions trade-offs
- **Build schedules:** Year-by-year plant construction recommendations
- **Capacity mix:** Evolution of generation portfolio over 20 years
- **Cost breakdown:** Capital, operating, maintenance for each scenario

### 2. Visualizations
- Pareto frontier plot (cost vs emissions)
- Capacity expansion timeline
- Generation mix evolution
- Cost and emissions trajectories
- Sensitivity analysis charts

### 3. Policy Insights
- Marginal cost of carbon reduction ($/ton CO2)
- Trade-offs between cost minimization and emissions goals
- Impact of different assumptions (discount rate, demand growth, technology costs)
- Recommendations for Ontario policymakers

## Success Criteria

✅ **Model correctly implements all constraints and objectives**
✅ **Generates valid Pareto frontier with multiple solutions**
✅ **Results are interpretable and actionable**
✅ **Code is well-documented and tested**
✅ **Visualizations clearly communicate findings**
✅ **Sensitivity analysis shows robustness of recommendations**

## Development Tips

### Start Simple, Build Complexity
1. Get a simple model working first (single objective, few constraints)
2. Add complexity incrementally (multi-objective, ramp rates, etc.)
3. Test at each stage before proceeding

### Debugging Strategies
- Start with small problem (2-3 years, 3 plant types)
- Check constraint satisfaction explicitly
- Verify objective function calculations manually
- Compare results with intuition/known solutions

### Performance Optimization
- Use continuous variables first, switch to integer later if needed
- Exploit problem structure (e.g., time-decomposable constraints)
- Set appropriate solver tolerances
- Use warm starts when generating Pareto frontier

### Model Validation
- Verify that all constraints are satisfied
- Check that solutions make physical sense
- Compare with literature/existing studies
- Test edge cases (zero cost constraint, zero emissions constraint)

## Extensions & Future Work

**Potential Enhancements:**
1. **Battery storage** - Add energy storage as a technology option
2. **Hourly resolution** - Model at hourly rather than annual level
3. **Uncertainty** - Add stochastic optimization for uncertain demand
4. **Transmission** - Model transmission constraints between regions
5. **Imports/exports** - Allow electricity trading with Quebec/US
6. **Environmental constraints** - Water usage, land use, etc.
7. **Reliability** - Model LOLP (Loss of Load Probability)
8. **Seasonal storage** - Hydrogen, pumped hydro
9. **Demand response** - Allow flexible demand
10. **Carbon pricing** - Model impact of carbon tax/cap-and-trade

## Resources & References

### Ontario Energy System
- IESO (Independent Electricity System Operator): https://www.ieso.ca
- Ontario Power Generation: https://www.opg.com
- Canada Energy Regulator: https://www.cer-rec.gc.ca

### Optimization & Modeling
- Pyomo Documentation: https://pyomo.readthedocs.io
- HiGHS Solver: https://highs.dev
- Capacity Expansion Planning: Review literature on "generation expansion planning"

### Similar Projects
- NREL ReEDS model (US-based)
- TIMES energy system model
- OSeMOSYS (Open Source Energy Modeling System)

## Getting Started Command Sequence

```bash
# 1. Create project directory
mkdir ontario-power-optimization
cd ontario-power-optimization

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install pyomo highspy numpy pandas matplotlib seaborn plotly tqdm pyyaml pytest jupyter

# 4. Create project structure
mkdir -p data/{raw,processed} src/{optimization,solvers,analysis,utils} notebooks tests results/{figures,data,reports} scripts

# 5. Copy data file
# Copy Ontario_Energy_Data_Summary.md to data/raw/

# 6. Start with Phase 1
# Create data loading utilities and parameter files

# 7. Open Jupyter notebook for prototyping
jupyter notebook notebooks/01_data_exploration.ipynb
```

## Example Usage

```python
# Load the optimization model
from src.optimization.model import PowerSystemOptimization
from src.analysis.pareto import generate_pareto_frontier

# Create model instance
model = PowerSystemOptimization(
    start_year=2025,
    end_year=2045,
    data_path='data/processed/'
)

# Run single objective optimization (minimize cost)
cost_optimal = model.optimize(objective='cost')
print(f"Minimum cost: ${cost_optimal.total_cost/1e9:.2f}B")
print(f"Total emissions: {cost_optimal.total_emissions/1e6:.2f}M tons")

# Generate Pareto frontier
pareto_solutions = generate_pareto_frontier(
    model=model,
    n_points=15,
    method='weighted_sum'
)

# Visualize results
from src.analysis.visualizations import plot_pareto_frontier
plot_pareto_frontier(pareto_solutions, save_path='results/figures/pareto.png')

# Export results
pareto_solutions.to_csv('results/data/pareto_solutions.csv')
```

## Contact & Support

For questions or issues:
- Review Ontario_Energy_Data_Summary.md for data details
- Check Pyomo documentation for modeling questions
- Use Claude Code for implementation assistance

---

**Ready to build! Start with Phase 1: Setup & Data Preparation**
