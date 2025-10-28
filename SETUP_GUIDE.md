# Setup Guide

## Quick Start

### 1. Install Dependencies

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

### 2. Generate Sample Data

```bash
# This creates the three required data files in data/processed/
python data/load_data.py
```

Expected output:
```
✓ Sample data files created successfully
✓ Data validation passed
```

This creates:
- `data/processed/plant_parameters.json` - All plant technical and cost parameters
- `data/processed/initial_capacity.json` - 2025 starting capacities
- `data/processed/demand_forecast.csv` - 20-year demand projections

### 3. Explore the Data

```bash
# Launch Jupyter notebook
jupyter notebook notebooks/01_data_exploration.ipynb
```

This notebook will:
- Load and display all data
- Visualize plant parameters
- Plot demand forecasts
- Show initial capacity mix
- Validate data consistency

## Project Structure

```
ontario-power-optimization/
├── data/
│   ├── raw/                           # Raw data sources (reference only)
│   ├── processed/                     # Generated data files (JSON, CSV)
│   │   ├── plant_parameters.json     # Created by load_data.py
│   │   ├── initial_capacity.json     # Created by load_data.py
│   │   └── demand_forecast.csv       # Created by load_data.py
│   └── load_data.py                  # Data generation script
│
├── src/
│   ├── optimization/                  # Core optimization model
│   │   ├── model.py                  # Main PowerSystemOptimization class
│   │   ├── variables.py              # Decision variables (to implement)
│   │   ├── objectives.py             # Cost & emissions objectives (to implement)
│   │   └── constraints.py            # All constraints (to implement)
│   │
│   ├── solvers/                       # Solver interfaces
│   │   ├── base_solver.py            # Abstract base class
│   │   ├── highs_solver.py           # HiGHS (open-source)
│   │   └── gurobi_solver.py          # Gurobi (commercial)
│   │
│   ├── analysis/                      # Analysis tools
│   │   ├── pareto.py                 # Pareto frontier generation (to implement)
│   │   ├── visualizations.py         # Plotting functions (to implement)
│   │   └── sensitivity.py            # Sensitivity analysis (to implement)
│   │
│   └── utils/                         # Utilities
│       ├── financial.py              # NPV, LCOE calculations
│       └── time_series.py            # Demand profile generation
│
├── scripts/                           # Execution scripts
│   ├── run_optimization.py           # Main optimization runner (to implement)
│   ├── generate_pareto.py            # Pareto frontier generator (to implement)
│   └── create_report.py              # Report generator (to implement)
│
├── notebooks/                         # Jupyter notebooks
│   └── 01_data_exploration.ipynb     # Data exploration notebook
│
├── tests/                             # Unit tests (to implement)
│
└── results/                           # Output files
    ├── figures/                       # Plots and visualizations
    ├── data/                          # Result datasets
    └── reports/                       # Summary reports
```

## Next Steps

Following the implementation plan in `Claude.md`:

### Phase 1: ✅ Setup & Data Preparation (DONE)
- [x] Project structure created
- [x] Dependencies configured
- [x] Data loading utilities written
- [x] Sample data generated

### Phase 2: Prototype Model (NEXT)
Build a simplified 5-year model to validate the approach:

1. **Implement decision variables** in `src/optimization/variables.py`
   - `x[t,i]`: Plants to build
   - `p[t,i]`: Power generation
   - `N[t,i]`: Total capacity

2. **Implement cost objective** in `src/optimization/objectives.py`
   - Capital costs (CAPEX)
   - Operating costs (OPEX)
   - Maintenance costs

3. **Implement basic constraints** in `src/optimization/constraints.py`
   - Demand satisfaction
   - Capacity constraints
   - Reserve margin

4. **Complete main model** in `src/optimization/model.py`
   - Build Pyomo model
   - Connect to solver
   - Extract results

5. **Test the prototype**
   ```bash
   python scripts/run_optimization.py --objective cost --solver highs
   ```

### Phase 3: Full Single-Objective Model
- Extend to 20 years
- Add construction lead times
- Add ramp rate constraints
- Add retirement logic

### Phase 4: Multi-Objective Optimization
- Implement emissions objective
- Generate Pareto frontier
- Compare trade-offs

### Phase 5: Analysis & Visualization
- Create comprehensive plots
- Implement sensitivity analysis
- Export results

### Phase 6: Documentation & Testing
- Write unit tests
- Add comprehensive docstrings
- Create example notebooks

## Testing the Setup

```bash
# Test Python imports
python -c "import pyomo.environ as pyo; print('Pyomo OK')"
python -c "import highspy; print('HiGHS OK')"
python -c "import pandas; print('Pandas OK')"

# Test data loading
python data/load_data.py

# Verify file structure
ls -la data/processed/
```

## Troubleshooting

### HiGHS solver not found
```bash
pip install --upgrade highspy
```

### Pyomo issues
```bash
pip install --upgrade pyomo
```

### Jupyter kernel not found
```bash
python -m ipykernel install --user --name=venv
```

## Development Workflow

1. **Start with notebooks** for prototyping
2. **Move working code** to `src/` modules
3. **Test** with small problems first
4. **Scale up** gradually
5. **Document** as you go

## Resources

- **Project plan**: `Claude.md`
- **Data reference**: `data/raw/Ontario_Energy_Data_Summary.md`
- **Pyomo docs**: https://pyomo.readthedocs.io
- **HiGHS docs**: https://highs.dev

## Ready to Code!

You're all set up. Start with Phase 2 by implementing the prototype model:

```bash
# Open the model file
code src/optimization/model.py

# Or use a notebook
jupyter notebook notebooks/02_prototype_model.ipynb
```

Good luck! 🚀
