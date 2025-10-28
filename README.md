# Ontario Power Plant Optimization

A multi-objective optimization model to determine the optimal mix of power plants Ontario should construct over the next 20 years (2025-2045) to meet growing electricity demand while minimizing costs and carbon emissions.

## Overview

Ontario's electricity demand is projected to grow 75% by 2050 (from 151 TWh to 263 TWh annually). This project optimizes:

**Primary Objectives:**
1. Minimize Total System Cost (capital + operating + maintenance)
2. Minimize Carbon Emissions

**Constraints:**
- Meet hourly electricity demand
- Maintain 15% reserve margin above peak demand
- Respect ramp rate limits
- Account for plant construction lead times and lifespans

**Plant Types:** Nuclear, Wind, Solar, Natural Gas, Hydro, Biofuel

## Quick Start

### Installation

```bash
# Clone the repository
cd ontario-power-optimization

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Generate Sample Data

```bash
# Create sample data files
python data/load_data.py
```

### Run Optimization

```python
from src.optimization.model import PowerSystemOptimization
from src.analysis.pareto import generate_pareto_frontier

# Create model instance
model = PowerSystemOptimization(
    start_year=2025,
    end_year=2045,
    data_path='data/processed/'
)

# Run optimization
results = model.optimize(objective='cost', solver='highs')

# Generate Pareto frontier
pareto = generate_pareto_frontier(model, n_points=15)
```

## Project Structure

```
ontario-power-optimization/
├── data/                          # Data files
│   ├── raw/                       # Raw data sources
│   ├── processed/                 # Processed data (JSON, CSV)
│   └── load_data.py              # Data loading utilities
├── src/                           # Source code
│   ├── optimization/              # Optimization model
│   │   ├── model.py              # Main model class
│   │   ├── variables.py          # Decision variables
│   │   ├── objectives.py         # Objective functions
│   │   └── constraints.py        # Constraints
│   ├── solvers/                   # Solver interfaces
│   │   ├── base_solver.py        # Abstract base
│   │   ├── highs_solver.py       # HiGHS solver
│   │   └── gurobi_solver.py      # Gurobi solver
│   ├── analysis/                  # Analysis tools
│   │   ├── pareto.py             # Pareto frontier
│   │   ├── visualizations.py     # Plotting
│   │   └── sensitivity.py        # Sensitivity analysis
│   └── utils/                     # Utilities
│       ├── financial.py          # NPV, LCOE calculations
│       └── time_series.py        # Demand profiles
├── notebooks/                     # Jupyter notebooks
├── tests/                         # Unit tests
├── results/                       # Output files
│   ├── figures/                  # Plots
│   ├── data/                     # Result datasets
│   └── reports/                  # Reports
└── scripts/                       # Execution scripts
```

## Features

### Optimization Model
- Multi-objective optimization (cost vs emissions)
- 20-year planning horizon (2025-2045)
- 6 plant types with realistic parameters
- Construction lead times and plant lifespans
- Ramp rate constraints
- Reserve margin requirements

### Analysis Tools
- Pareto frontier generation
- Capacity expansion visualization
- Cost breakdown analysis
- Emissions trajectory tracking
- Sensitivity analysis (discount rate, demand growth, technology costs)

### Solver Support
- **HiGHS**: Open-source, fast, no license required (recommended for prototyping)
- **Gurobi**: Commercial solver with academic licenses (recommended for production)
- **CPLEX**: IBM commercial solver

## Key Parameters

### Plant Data
- **Capital costs**: $1,300-17,500/kW
- **Operating costs**: $10-55/MWh
- **Capacity factors**: 15-90%
- **Emission factors**: 0.011-0.45 tons CO2/MWh
- **Construction lead times**: 2-7 years
- **Plant lifespans**: 25-100 years

### System Parameters
- **Discount rate**: 3.92% (real)
- **Demand growth**: 2.2% annually
- **Reserve margin**: 15%
- **Planning horizon**: 2025-2045

## Development Phases

The project follows a structured development approach:

1. **Phase 1**: Setup & Data Preparation
2. **Phase 2**: Prototype Model (5-year, simplified)
3. **Phase 3**: Full Single-Objective Model (20-year)
4. **Phase 4**: Multi-Objective Optimization
5. **Phase 5**: Analysis & Visualization
6. **Phase 6**: Documentation & Testing

See `Claude.md` for detailed implementation plan.

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_model.py

# Run with coverage
pytest --cov=src tests/
```

## Results

The model generates:

1. **Pareto Frontier**: 10-20 optimal solutions showing cost-emissions trade-offs
2. **Build Schedules**: Year-by-year plant construction recommendations
3. **Capacity Mix**: Evolution of generation portfolio over 20 years
4. **Cost Analysis**: Breakdown by capital, operating, and maintenance
5. **Emissions Trajectories**: Carbon emissions over time
6. **Sensitivity Analysis**: Impact of parameter variations

## Documentation

- `Claude.md`: Detailed project plan and implementation guide
- `data/raw/Ontario_Energy_Data_Summary.md`: Reference data sources
- Docstrings: All functions include comprehensive documentation

## Contributing

This is an academic project. For questions or improvements, please refer to the project documentation.

## License

This project is for educational and research purposes.

## Acknowledgments

Data sources:
- IESO (Independent Electricity System Operator)
- Ontario Power Generation
- Canada Energy Regulator

Technologies:
- Pyomo (optimization modeling)
- HiGHS (open-source solver)
- Python scientific stack (NumPy, Pandas, Matplotlib)

## Contact

For questions about this project, refer to the documentation in `Claude.md` or the data summary in `data/raw/Ontario_Energy_Data_Summary.md`.
