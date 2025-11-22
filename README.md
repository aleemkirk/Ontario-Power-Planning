# Ontario Power Plant Optimization

A multi-objective linear optimization model to determine the optimal mix of power plants Ontario should construct over the next 20 years (2025-2045) to meet growing electricity demand while minimizing costs and carbon emissions.

## Overview

Ontario's electricity demand is projected to grow 75% by 2050 (from 151 TWh to 263 TWh annually). This project optimizes:

**Primary Objectives:**
1. Minimize Total System Cost (capital + operating + maintenance)
2. Minimize Carbon Emissions

**Plant Types:** Nuclear, Wind, Solar, Natural Gas, Hydro, Biofuel

## Quick Start

### Installation

```bash
cd ontario-power-optimization

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Optimization

```bash
python generate_pareto_hourly.py
```

This generates a Pareto frontier showing cost-emissions trade-offs and saves results to `results/data/`.

## Project Structure

```
ontario-power-optimization/
├── generate_pareto_hourly.py      # Main entry point
├── data/
│   ├── processed/                 # Plant parameters, demand forecasts
│   ├── raw/ieso/                  # Raw IESO demand data
│   └── create_representative_days.py
├── src/
│   ├── optimization/
│   │   ├── model_hourly.py        # Main optimization model
│   │   ├── objectives.py          # Cost & emissions objectives
│   │   ├── constraints.py         # Reserve margin, capacity dynamics
│   │   ├── constraints_hourly.py  # Hourly demand, ramp rates
│   │   └── constraints_nuclear_policy.py
│   └── utils/
│       ├── financial.py           # NPV calculations
│       └── load_hourly_data.py    # Hourly demand data loading
├── scripts/
│   ├── download_ieso_data.py      # Download raw demand data
│   └── process_ieso_data.py       # Process raw data
└── results/
    ├── data/                      # Output CSV/JSON files
    └── figures/                   # Generated plots
```

## Linear Model Formulation

### Sets

| Set | Description |
|-----|-------------|
| T | Years in planning horizon (2025-2045) |
| I | Plant types {nuclear, wind, solar, gas, hydro, biofuel} |
| D | Representative days (12 clusters) |
| H | Hours per day (0-23) |

### Decision Variables

| Variable | Domain | Description |
|----------|--------|-------------|
| x[t,i] | R+ | New capacity of plant type i to build in year t (MW) |
| N[t,i] | R+ | Total operating capacity of plant type i in year t (MW) |
| p[t,d,h,i] | R+ | Power output from plant type i in year t, day d, hour h (MW) |
| v[t,d,h,i] | R+ | Ramp rate violation slack variable (MW) |

### Parameters

| Parameter | Description |
|-----------|-------------|
| CapEx[i] | Capital cost for plant type i ($/kW) |
| OpEx[i] | Operating cost for plant type i ($/MWh) |
| MaintEx[i] | Annual maintenance cost for plant type i ($/MW/year) |
| CF[i] | Capacity factor for plant type i |
| EF[i] | Emission factor for plant type i (tons CO2/MWh) |
| RR[i] | Ramp rate for plant type i (MW/min per MW capacity) |
| LT[i] | Construction lead time for plant type i (years) |
| W[d] | Weight of representative day d (number of days it represents) |
| Demand[t,d,h] | Electricity demand in year t, day d, hour h (MW) |
| PeakDemand[t] | Peak demand in year t (MW) |
| InitCap[i] | Initial capacity of plant type i in 2025 (MW) |
| r | Discount rate (3.92%) |
| RM | Reserve margin (15%) |
| RampPenalty | Penalty for ramp violations ($/MW) |

### Objective Functions

**1. Minimize Total Cost (NPV)**

```
Z_cost = CapitalCost + OperatingCost + MaintenanceCost + RampPenalty

where:
  CapitalCost    = sum over t,i of: x[t,i] * CapEx[i] * 1000 / (1+r)^(t-2025)
  OperatingCost  = sum over t,d,h,i of: p[t,d,h,i] * W[d] * OpEx[i] / (1+r)^(t-2025)
  MaintenanceCost = sum over t,i of: N[t,i] * MaintEx[i] / (1+r)^(t-2025)
  RampPenalty    = sum over t,d,h,i of: v[t,d,h,i] * W[d] * RampPenalty / (1+r)^(t-2025)
```

**2. Minimize Total Emissions**

```
Z_emissions = sum over t,d,h,i of: p[t,d,h,i] * W[d] * EF[i]
```

**3. Multi-Objective (Weighted Sum)**

```
Z_multi = alpha * (Z_cost / Z_cost_max) + (1 - alpha) * (Z_emissions / Z_emissions_max)

where alpha in [0, 1]:
  - alpha = 1.0: Minimize cost only
  - alpha = 0.0: Minimize emissions only
  - alpha in (0,1): Trade-off between objectives
```

### Constraints

**1. Hourly Demand Satisfaction**

Total generation must meet demand every hour:
```
sum over i of: p[t,d,h,i] >= Demand[t,d,h]    for all t, d, h
```

**2. Hourly Capacity Limit**

Generation cannot exceed available capacity:
```
p[t,d,h,i] <= N[t,i] * CF[i]    for all t, d, h, i
```

**3. Reserve Margin**

Total capacity must exceed peak demand by reserve margin:
```
sum over i of: N[t,i] >= (1 + RM) * PeakDemand[t]    for all t
```

**4. Ramp Rate Constraints (Soft)**

Limit rate of change between consecutive hours:
```
p[t,d,h,i] - p[t,d,h-1,i] <= RR[i] * N[t,i] * 60 + v[t,d,h,i]    for all t, d, h>0, i
p[t,d,h-1,i] - p[t,d,h,i] <= RR[i] * N[t,i] * 60 + v[t,d,h,i]    for all t, d, h>0, i
```

**5. Capacity Dynamics (with Lead Times)**

Track capacity evolution accounting for construction lead times:
```
N[2025,i] = InitCap[i]    for all i (initial condition)

N[t,i] = N[t-1,i] + x[t-LT[i],i]    for all t > 2025, i
         (new capacity comes online LT[i] years after construction starts)
```

**6. Non-Negativity**
```
x[t,i] >= 0    for all t, i
N[t,i] >= 0    for all t, i
p[t,d,h,i] >= 0    for all t, d, h, i
v[t,d,h,i] >= 0    for all t, d, h, i
```

## Plant Parameters

| Plant Type | CapEx ($/kW) | OpEx ($/MWh) | Capacity Factor | Emissions (tCO2/MWh) | Ramp Rate (MW/min/MW) | Lead Time (yrs) |
|------------|--------------|--------------|-----------------|----------------------|-----------------------|-----------------|
| Nuclear | 17,500 | 22 | 90% | 0.012 | 0.02 | 7 |
| Wind | 1,900 | 12 | 35% | 0.011 | 0.05 | 2 |
| Solar | 1,300 | 12 | 15% | 0.048 | 0.10 | 2 |
| Gas | 1,500 | 55 | 55% | 0.450 | 0.04 | 3 |
| Hydro | 3,000 | 10 | 50% | 0.024 | 0.15 | 6 |
| Biofuel | 3,200 | 42 | 80% | 0.230 | 0.01 | 3 |

## System Parameters

| Parameter | Value |
|-----------|-------|
| Planning horizon | 2025-2045 (21 years) |
| Discount rate | 3.92% (real) |
| Reserve margin | 15% |
| Representative days | 12 (clustered from historical data) |
| Hours per day | 24 |
| Ramp penalty | $1,000/MW |

## Results

The model generates:

1. **Pareto Frontier**: Trade-off curve between cost and emissions
2. **Capacity Mix**: Optimal generation portfolio for each scenario
3. **Build Schedule**: Year-by-year construction plan
4. **Marginal Cost of Carbon**: $/ton CO2 to reduce emissions

Output files in `results/data/`:
- `pareto_frontier_hourly.csv` - Summary of all Pareto points
- `pareto_capacity_mix_hourly.csv` - Capacity by plant type for each solution
- `pareto_frontier_hourly_detailed.json` - Full results

## Solver

The model uses **HiGHS** (open-source LP/MIP solver) via Pyomo. Gurobi is also supported for faster performance on large problems.

## Data Sources

- **IESO** (Independent Electricity System Operator) - Historical hourly demand data
- **Ontario Power Generation** - Plant capacity data
- **Canada Energy Regulator** - Cost and technology parameters

## License

This project is for educational and research purposes.
