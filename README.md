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
| $\mathcal{T}$ | Years in planning horizon {2025, ..., 2045} |
| $\mathcal{I}$ | Plant types {nuclear, wind, solar, gas, hydro, biofuel} |
| $\mathcal{D}$ | Representative days {0, ..., 11} |
| $\mathcal{H}$ | Hours per day {0, ..., 23} |

### Decision Variables

| Variable | Domain | Description |
|----------|--------|-------------|
| $x_{t,i}$ | $\mathbb{R}^+$ | New capacity of plant type $i$ to build in year $t$ (MW) |
| $N_{t,i}$ | $\mathbb{R}^+$ | Total operating capacity of plant type $i$ in year $t$ (MW) |
| $p_{t,d,h,i}$ | $\mathbb{R}^+$ | Power output from plant type $i$ in year $t$, day $d$, hour $h$ (MW) |
| $v_{t,d,h,i}$ | $\mathbb{R}^+$ | Ramp rate violation slack variable (MW) |

### Parameters

| Parameter | Description |
|-----------|-------------|
| $C^{cap}_i$ | Capital cost for plant type $i$ ($/kW) |
| $C^{op}_i$ | Operating cost for plant type $i$ ($/MWh) |
| $C^{maint}_i$ | Annual maintenance cost for plant type $i$ ($/MW/year) |
| $CF_i$ | Capacity factor for plant type $i$ |
| $EF_i$ | Emission factor for plant type $i$ (tons CO₂/MWh) |
| $RR_i$ | Ramp rate for plant type $i$ (MW/min per MW capacity) |
| $LT_i$ | Construction lead time for plant type $i$ (years) |
| $w_d$ | Weight of representative day $d$ (number of days it represents) |
| $D_{t,d,h}$ | Electricity demand in year $t$, day $d$, hour $h$ (MW) |
| $D^{peak}_t$ | Peak demand in year $t$ (MW) |
| $N^{init}_i$ | Initial capacity of plant type $i$ in 2025 (MW) |
| $r$ | Discount rate (3.92%) |
| $RM$ | Reserve margin (15%) |
| $\rho$ | Penalty for ramp violations ($/MW) |

### Objective Functions

**1. Minimize Total Cost (NPV)**

$$Z_{cost} = Z_{capex} + Z_{opex} + Z_{maint} + Z_{ramp}$$

where:

$$Z_{capex} = \sum_{t \in \mathcal{T}} \sum_{i \in \mathcal{I}} \frac{x_{t,i} \cdot C^{cap}_i \cdot 1000}{(1+r)^{t-2025}}$$

$$Z_{opex} = \sum_{t \in \mathcal{T}} \sum_{d \in \mathcal{D}} \sum_{h \in \mathcal{H}} \sum_{i \in \mathcal{I}} \frac{p_{t,d,h,i} \cdot w_d \cdot C^{op}_i}{(1+r)^{t-2025}}$$

$$Z_{maint} = \sum_{t \in \mathcal{T}} \sum_{i \in \mathcal{I}} \frac{N_{t,i} \cdot C^{maint}_i}{(1+r)^{t-2025}}$$

$$Z_{ramp} = \sum_{t \in \mathcal{T}} \sum_{d \in \mathcal{D}} \sum_{h \in \mathcal{H}} \sum_{i \in \mathcal{I}} \frac{v_{t,d,h,i} \cdot w_d \cdot \rho}{(1+r)^{t-2025}}$$

**2. Minimize Total Emissions**

$$Z_{emissions} = \sum_{t \in \mathcal{T}} \sum_{d \in \mathcal{D}} \sum_{h \in \mathcal{H}} \sum_{i \in \mathcal{I}} p_{t,d,h,i} \cdot w_d \cdot EF_i$$

**3. Multi-Objective (Weighted Sum)**

$$Z_{multi} = \alpha \cdot \frac{Z_{cost}}{Z^{max}_{cost}} + (1 - \alpha) \cdot \frac{Z_{emissions}}{Z^{max}_{emissions}}$$

where $\alpha \in [0, 1]$:
- $\alpha = 1$: Minimize cost only
- $\alpha = 0$: Minimize emissions only
- $\alpha \in (0,1)$: Trade-off between objectives

### Constraints

**1. Hourly Demand Satisfaction**

Total generation must meet demand every hour:

$$\sum_{i \in \mathcal{I}} p_{t,d,h,i} \geq D_{t,d,h} \quad \forall t \in \mathcal{T}, d \in \mathcal{D}, h \in \mathcal{H}$$

**2. Hourly Capacity Limit**

Generation cannot exceed available capacity:

$$p_{t,d,h,i} \leq N_{t,i} \cdot CF_i \quad \forall t \in \mathcal{T}, d \in \mathcal{D}, h \in \mathcal{H}, i \in \mathcal{I}$$

**3. Reserve Margin**

Total capacity must exceed peak demand by reserve margin:

$$\sum_{i \in \mathcal{I}} N_{t,i} \geq (1 + RM) \cdot D^{peak}_t \quad \forall t \in \mathcal{T}$$

**4. Ramp Rate Constraints (Soft)**

Limit rate of change between consecutive hours:

$$p_{t,d,h,i} - p_{t,d,h-1,i} \leq RR_i \cdot N_{t,i} \cdot 60 + v_{t,d,h,i} \quad \forall t, d, h > 0, i$$

$$p_{t,d,h-1,i} - p_{t,d,h,i} \leq RR_i \cdot N_{t,i} \cdot 60 + v_{t,d,h,i} \quad \forall t, d, h > 0, i$$

**5. Capacity Dynamics (with Lead Times)**

Initial condition:

$$N_{2025,i} = N^{init}_i \quad \forall i \in \mathcal{I}$$

Capacity evolution (new capacity comes online $LT_i$ years after construction starts):

$$N_{t,i} = N_{t-1,i} + x_{t-LT_i,i} \quad \forall t > 2025, i \in \mathcal{I}$$

**6. Non-Negativity**

$$x_{t,i} \geq 0, \quad N_{t,i} \geq 0, \quad p_{t,d,h,i} \geq 0, \quad v_{t,d,h,i} \geq 0$$

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
