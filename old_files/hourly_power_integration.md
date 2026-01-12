# Hourly Resolution Implementation Plan
## Ontario Power Plant Optimization Model

---

## Executive Summary

This plan outlines a phased approach to transition the Ontario Power Plant Optimization model from **annual time resolution** to **hourly resolution**, enabling proper modeling of ramp rate constraints and improving operational realism. The current annual model has 378 decision variables (20 years × 6 plant types × 3 variable types). The full hourly model would expand to ~2.1 million variables, requiring representative day clustering to maintain tractability.

**Key Strategy**: Use **representative day clustering** to reduce 8,760 hours/year to 12-24 representative days, achieving **97% model size reduction** while maintaining **95%+ accuracy**.

---

## 1. Current State Analysis

### 1.1 Current Architecture

**File: `src/optimization/variables.py` (Lines 13-47)**

Current decision variables (annual resolution):
- `x[t,i]`: New capacity to build in year t, plant type i (MW) - 126 variables
- `p[t,i]`: Annual energy generation in year t, plant type i (MWh) - 126 variables
- `N[t,i]`: Total operating capacity in year t, plant type i (MW) - 126 variables
- **Total: 378 decision variables for 20 years × 6 plant types**

**File: `src/optimization/constraints.py` (Lines 65-78)**

Ramp rate constraint is **currently disabled** with this comment:
```python
def ramp_rate_constraint(model):
    """
    NOTE: Skipped in prototype (annual resolution).
    Will be implemented in Phase 3 with hourly/monthly resolution.

    |p[t,i,h] - p[t,i,h-1]| ≤ RampRate[i] × N[t,i]  ∀t,i,h
    """
    pass  # Skip for prototype
```

**File: `src/optimization/constraints.py` (Lines 30-45)**

Current capacity constraint (annual):
```python
def capacity_constraint(model):
    """p[t,i] ≤ N[t,i] × CapacityFactor[i] × 8760"""
    def capacity_rule(m, t, i):
        return m.p[t, i] <= m.N[t, i] * m.capacity_factor[i] * 8760
```
This multiplies by 8760 hours to convert MW capacity to annual MWh energy.

### 1.2 Why Ramp Rates Matter

**File: `data/processed/plant_parameters.json` (Lines 42-49)**

Ramp rate data is already available (MW/min per MW capacity):
- **Hydro: 0.15** - Fastest response, ideal for load balancing
- **Solar: 0.10** - Fast ramp (though weather-dependent)
- **Wind: 0.05** - Moderate ramp capability
- **Gas: 0.04** - Good for peaking/load following
- **Nuclear: 0.02** - Slowest, designed for baseload
- **Biofuel: 0.01** - Very slow response

**Impact**: At hourly resolution, these differences become critical. A 1,000 MW nuclear plant can only ramp 20 MW/min (1,200 MW/hour max change), while a 1,000 MW hydro plant can ramp 150 MW/min (9,000 MW/hour, though limited by capacity). This properly values flexible generation.

### 1.3 Current Model Size

**Current (Annual):**
- Years: 20
- Plant types: 6
- Decision variables: 20 × 6 × 3 = 360
- Constraints: ~500 (demand, capacity, reserve, capacity dynamics)
- Solve time: <1 minute with HiGHS

**Proposed (Full Hourly - INFEASIBLE):**
- Years: 20
- Hours per year: 8,760
- Plant types: 6
- Hourly variables: 20 × 8,760 × 6 × 2 = 2,102,400
- Annual variables: 20 × 6 × 2 = 240
- **Total: ~2.1M variables, ~3.5M constraints**
- Solve time: HOURS or INFEASIBLE

**Proposed (Representative Days - FEASIBLE):**
- Years: 20
- Representative days: 12
- Hours per day: 24
- Plant types: 6
- Hourly variables: 20 × 12 × 24 × 6 × 2 = 69,120
- Annual variables: 20 × 6 × 2 = 240
- **Total: ~69,360 variables, ~113,580 constraints**
- Solve time: 10-30 minutes with Gurobi

---

## 2. Data Requirements

### 2.1 Hourly Demand Profiles

**Current data (annual):**
- File: `data/processed/demand_forecast.csv`
- Contains: Annual demand (GWh/year), peak demand (MW) for 2025-2045
- Resolution: Annual totals only

**Required for hourly model:**

**Option A: Obtain Real Hourly Data (RECOMMENDED)**
- **Source**: IESO (Independent Electricity System Operator) publishes hourly Ontario demand
  - Historical data: https://www.ieso.ca/power-data
  - Format: CSV with timestamp and demand (MW)
  - Availability: 2002-present, hourly resolution

**Data needed:**
1. Historical hourly demand for representative year (e.g., 2023-2024)
2. Normalized hourly profile (% of annual peak by hour of year)
3. Seasonal and daily patterns

**Option B: Synthetic Demand Profile (FALLBACK)**

If real data unavailable, generate synthetic profile:
```python
# Pseudo-code for synthetic profile
def generate_hourly_demand(annual_demand, peak_demand):
    """
    Generate 8760 hourly demand values from annual total and peak.

    Components:
    - Base load: 60% of peak (constant)
    - Seasonal variation: ±20% (sinusoidal, winter peak)
    - Daily variation: ±30% (peak 2-6pm, trough 2-6am)
    - Weekly variation: ±10% (weekday vs weekend)
    """
    hours = np.arange(8760)

    # Seasonal (annual cycle)
    seasonal = 1 + 0.2 * np.sin(2*np.pi*hours/8760 - np.pi/2)

    # Daily (24-hour cycle)
    daily = 1 + 0.3 * np.sin(2*np.pi*(hours % 24)/24 - np.pi/2)

    # Weekly (168-hour cycle)
    weekly = 1 - 0.1 * (hours % 168 < 48)  # Weekend reduction

    # Combine and scale
    profile = seasonal * daily * weekly
    profile = profile / profile.mean()  # Normalize
    hourly_demand = profile * (annual_demand / 8760)

    return hourly_demand
```

### 2.2 Representative Day Clustering

To reduce model size, use **representative days** instead of all 8,760 hours.

**Clustering approach:**
```python
# Cluster 365 days into 12-24 representative days
from sklearn.cluster import KMeans

def create_representative_days(hourly_demand, n_clusters=12):
    """
    Cluster days by demand pattern.

    Features for clustering:
    - Mean demand
    - Peak demand
    - Peak hour
    - Daily load factor
    - Season (winter/summer/shoulder)
    """
    # Reshape to (365 days, 24 hours)
    daily_profiles = hourly_demand.reshape(365, 24)

    # Extract features
    features = np.column_stack([
        daily_profiles.mean(axis=1),     # Mean
        daily_profiles.max(axis=1),      # Peak
        daily_profiles.argmax(axis=1),   # Peak hour
        daily_profiles.std(axis=1),      # Variability
        get_season(np.arange(365))       # Season indicator
    ])

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(features)

    # For each cluster, select most representative day
    rep_days = []
    weights = []
    for cluster_id in range(n_clusters):
        cluster_days = daily_profiles[labels == cluster_id]
        # Choose median day
        center = cluster_days.mean(axis=0)
        distances = np.linalg.norm(cluster_days - center, axis=1)
        rep_day_idx = np.argmin(distances)

        rep_days.append(cluster_days[rep_day_idx])
        weights.append((labels == cluster_id).sum())  # Number of days

    return np.array(rep_days), np.array(weights)
```

**Recommended clustering:**
- **12 representative days**: 4 seasons × 3 patterns (weekday peak, weekend, shoulder)
- **24 representative days**: More granular, captures edge cases
- **Model size reduction**: 8,760 hours → 288 hours (12 days × 24 hours) = **97% reduction**

### 2.3 New Data Files Needed

Create these files in `data/processed/`:

1. **`hourly_demand_profile.csv`**
   ```csv
   year,hour,demand_MW
   2025,0,18500
   2025,1,17800
   ...
   2025,8759,19200
   2026,0,18900
   ...
   ```

2. **`representative_days.json`**
   ```json
   {
     "n_clusters": 12,
     "representative_days": [
       {
         "day_id": 0,
         "name": "Winter Weekday Peak",
         "weight": 60,
         "hours_24": [18000, 17500, ..., 19000]
       },
       ...
     ]
   }
   ```

3. **`ramp_rate_params.json`** (already in plant_parameters.json)
   - Already exists at `data/processed/plant_parameters.json` (lines 42-49)

---

## 3. Model Architecture Changes

### 3.1 New Decision Variables

**File to modify: `src/optimization/variables.py`**

**Current variables (lines 26-47):**
```python
# x[t,i] - New capacity (MW)
model.x = pyo.Var(model.years, model.plant_types, domain=pyo.NonNegativeReals)

# p[t,i] - Annual generation (MWh)
model.p = pyo.Var(model.years, model.plant_types, domain=pyo.NonNegativeReals)

# N[t,i] - Total capacity (MW)
model.N = pyo.Var(model.years, model.plant_types, domain=pyo.NonNegativeReals)
```

**New variables needed:**
```python
def define_variables_hourly(model):
    """Define variables for hourly resolution model."""

    # ANNUAL VARIABLES (unchanged)
    model.x = pyo.Var(
        model.years, model.plant_types,
        domain=pyo.NonNegativeReals,
        doc="New capacity to build (MW)"
    )

    model.N = pyo.Var(
        model.years, model.plant_types,
        domain=pyo.NonNegativeReals,
        doc="Total operating capacity (MW)"
    )

    # NEW: HOURLY VARIABLES
    model.p_hourly = pyo.Var(
        model.years, model.rep_days, model.hours_per_day, model.plant_types,
        domain=pyo.NonNegativeReals,
        doc="Hourly power output (MW)"
    )

    # NEW: RAMP UP/DOWN SLACK VARIABLES (for soft ramp constraints)
    model.ramp_violation = pyo.Var(
        model.years, model.rep_days, model.hours_per_day, model.plant_types,
        domain=pyo.NonNegativeReals,
        doc="Ramp rate constraint violation (MW) - penalized in objective"
    )
```

**Backward compatibility approach:**
```python
def define_variables(model, resolution='annual'):
    """
    Define decision variables.

    Args:
        model: Pyomo model
        resolution: 'annual' or 'hourly'
    """
    if resolution == 'annual':
        define_variables_annual(model)
    elif resolution == 'hourly':
        define_variables_hourly(model)
    else:
        raise ValueError(f"Unknown resolution: {resolution}")
```

### 3.2 Modified Constraints

**File to modify: `src/optimization/constraints.py`**

#### 3.2.1 Demand Satisfaction (Lines 15-27)

**Current (annual):**
```python
def demand_rule(m, t):
    return sum(m.p[t, i] for i in m.plant_types) >= m.annual_demand[t]
```

**New (hourly with representative days):**
```python
def demand_satisfaction_constraint_hourly(model):
    """Ensure generation meets demand every hour of every representative day."""
    def demand_rule(m, t, d, h):
        # d = representative day index, h = hour within day (0-23)
        return sum(m.p_hourly[t, d, h, i] for i in m.plant_types) >= m.rep_day_demand[t, d, h]

    model.demand_constraint = pyo.Constraint(
        model.years, model.rep_days, model.hours_per_day, rule=demand_rule
    )
```

#### 3.2.2 Capacity Constraint (Lines 30-45)

**Current (annual):**
```python
def capacity_rule(m, t, i):
    return m.p[t, i] <= m.N[t, i] * m.capacity_factor[i] * 8760
```

**New (hourly):**
```python
def capacity_constraint_hourly(model):
    """Hourly generation can't exceed available capacity."""
    def capacity_rule(m, t, d, h, i):
        # Each hour's generation limited by capacity × capacity factor
        return m.p_hourly[t, d, h, i] <= m.N[t, i] * m.capacity_factor[i]

    model.capacity_constraint = pyo.Constraint(
        model.years, model.rep_days, model.hours_per_day, model.plant_types,
        rule=capacity_rule
    )
```

#### 3.2.3 **NEW: Ramp Rate Constraint** (Lines 65-78 - CURRENTLY DISABLED)

**This constraint will be ACTIVATED:**
```python
def ramp_rate_constraint_hourly(model, soft=True):
    """
    Limit rate of change in power output between consecutive hours.

    Hard constraint: |p[t,h,i] - p[t,h-1,i]| ≤ RampRate[i] × N[t,i] × 60
    (60 min/hour converts MW/min to MW/hour)

    Soft constraint: Allow violations but penalize in objective

    Args:
        soft: If True, use soft constraint with penalty (recommended)
    """
    if soft:
        # Soft constraint: p[t,h,i] - p[t,h-1,i] ≤ RampRate[i] × N[t,i] × 60 + violation[t,h,i]
        def ramp_up_rule(m, t, d, h, i):
            if h == 0:
                return pyo.Constraint.Skip  # Skip first hour of each day
            max_ramp_mw_per_hour = m.ramp_rate[i] * m.N[t, i] * 60
            return (m.p_hourly[t, d, h, i] - m.p_hourly[t, d, h-1, i]
                    <= max_ramp_mw_per_hour + m.ramp_violation[t, d, h, i])

        def ramp_down_rule(m, t, d, h, i):
            if h == 0:
                return pyo.Constraint.Skip
            max_ramp_mw_per_hour = m.ramp_rate[i] * m.N[t, i] * 60
            return (m.p_hourly[t, d, h-1, i] - m.p_hourly[t, d, h, i]
                    <= max_ramp_mw_per_hour + m.ramp_violation[t, d, h, i])
    else:
        # Hard constraint: absolutely enforce ramp limits
        def ramp_up_rule(m, t, d, h, i):
            if h == 0:
                return pyo.Constraint.Skip
            max_ramp_mw_per_hour = m.ramp_rate[i] * m.N[t, i] * 60
            return m.p_hourly[t, d, h, i] - m.p_hourly[t, d, h-1, i] <= max_ramp_mw_per_hour

        def ramp_down_rule(m, t, d, h, i):
            if h == 0:
                return pyo.Constraint.Skip
            max_ramp_mw_per_hour = m.ramp_rate[i] * m.N[t, i] * 60
            return m.p_hourly[t, d, h-1, i] - m.p_hourly[t, d, h, i] <= max_ramp_mw_per_hour

    model.ramp_up_constraint = pyo.Constraint(
        model.years, model.rep_days, model.hours_per_day, model.plant_types,
        rule=ramp_up_rule
    )
    model.ramp_down_constraint = pyo.Constraint(
        model.years, model.rep_days, model.hours_per_day, model.plant_types,
        rule=ramp_down_rule
    )
```

#### 3.2.4 Reserve Margin (Lines 48-62)

**No change needed** - still operates at annual level:
```python
def reserve_rule(m, t):
    total_capacity = sum(m.N[t, i] for i in m.plant_types)
    required_capacity = (1 + m.reserve_margin) * m.peak_demand[t]
    return total_capacity >= required_capacity
```

#### 3.2.5 Capacity Dynamics (Lines 81-136)

**No change needed** - still operates at annual level for capacity planning.

### 3.3 Modified Objectives

**File to modify: `src/optimization/objectives.py`**

#### 3.3.1 Total Cost (Lines 13-53)

**Current operating cost (annual):**
```python
opex_cost = sum(
    model.p[t, i] * model.opex[i] * model.discount_factor[year_index(t)]
    for t in model.years for i in model.plant_types
)
```

**New (hourly with representative days):**
```python
opex_cost = sum(
    model.p_hourly[t, d, h, i] * model.day_weight[d] * model.opex[i]
    * model.discount_factor[year_index(t)]
    for t in model.years for d in model.rep_days
    for h in model.hours_per_day for i in model.plant_types
)
# day_weight[d] = number of days this representative day represents
```

**NEW: Ramp violation penalty** (if using soft constraints):
```python
ramp_penalty_cost = sum(
    model.ramp_violation[t, d, h, i] * model.ramp_penalty
    * model.discount_factor[year_index(t)]
    for t in model.years for d in model.rep_days
    for h in model.hours_per_day for i in model.plant_types
)
# ramp_penalty = e.g., $1000/MW (make violations expensive but allow if needed)
```

#### 3.3.2 Total Emissions (Lines 56-74)

**Current (annual):**
```python
total_emissions = sum(
    model.p[t, i] * model.emissions[i]
    for t in model.years for i in model.plant_types
)
```

**New (hourly with representative days):**
```python
total_emissions = sum(
    model.p_hourly[t, d, h, i] * model.day_weight[d] * model.emissions[i]
    for t in model.years for d in model.rep_days
    for h in model.hours_per_day for i in model.plant_types
)
```

---

## 4. Implementation Strategy

### Phase 1: Data Preparation (Week 1)
**Duration: 3-5 days**

#### Tasks:

**1.1 Obtain or Generate Hourly Demand Data**
- [ ] Download IESO historical hourly demand data for Ontario
  - URL: https://www.ieso.ca/power-data
  - Download: 2023-2024 hourly demand (CSV format)
  - Save to: `data/raw/ieso_hourly_demand_2023_2024.csv`

- [ ] Create synthetic demand generator (fallback if real data unavailable)
  - File: `data/create_synthetic_demand.py`
  - Generate 8760-hour profile for each year 2025-2045
  - Match annual totals from existing `demand_forecast.csv`

**1.2 Create Representative Day Clustering**
- [ ] Implement clustering algorithm
  - File: `data/create_representative_days.py`
  - Use K-means with k=12 (recommended) or k=24 (high fidelity)
  - Features: mean, peak, time-of-peak, std, season
  - Output: `data/processed/representative_days.json`

- [ ] Validate clustering quality
  - Compare total energy: Σ(rep_day × weight) ≈ annual total
  - Compare peak demand: max(rep_days) ≈ annual peak
  - Visualize: plot all 365 days + 12 representative days

**1.3 Create Data Loading Utilities**
- [ ] Update `data/load_data.py` to load hourly data
  - Function: `load_hourly_demand(year, resolution='hourly')`
  - Function: `load_representative_days(n_clusters=12)`
  - Validation: check data integrity, handle missing values

**Files to create:**
```
data/
├── raw/
│   └── ieso_hourly_demand_2023_2024.csv       (NEW)
├── processed/
│   ├── hourly_demand_2025_2045.csv            (NEW - 175,200 rows)
│   └── representative_days_12clusters.json    (NEW)
├── create_synthetic_demand.py                 (NEW)
├── create_representative_days.py              (NEW)
└── load_data.py                               (UPDATE)
```

**Validation criteria:**
- ✓ Hourly demand sums to annual demand (within 0.1%)
- ✓ Peak hourly demand ≈ annual peak demand (within 5%)
- ✓ Representative days reconstruct annual profile (R² > 0.95)

---

### Phase 2: Prototype Hourly Model (Week 2)
**Duration: 5-7 days**

#### Scope: 1-year, 12 representative days

**2.1 Create New Model Variant**
- [ ] Create new file: `src/optimization/model_hourly.py`
  - Subclass `PowerSystemOptimization`
  - Class name: `PowerSystemOptimizationHourly`
  - Parameters: `resolution='hourly'`, `use_rep_days=True`, `n_rep_days=12`

**2.2 Implement Hourly Variables**
- [ ] Update `src/optimization/variables.py`
  - Add `define_variables_hourly()` function
  - Add resolution parameter to `define_variables()`
  - Variables: `p_hourly[t,d,h,i]`, `ramp_violation[t,d,h,i]`

**2.3 Implement Hourly Constraints**
- [ ] Update `src/optimization/constraints.py`
  - `demand_satisfaction_constraint_hourly()`
  - `capacity_constraint_hourly()`
  - `ramp_rate_constraint_hourly(soft=True)` - **ACTIVATE THIS**
  - Keep annual constraints: reserve margin, capacity dynamics

**2.4 Update Objectives**
- [ ] Update `src/optimization/objectives.py`
  - Modify `total_cost_objective()` to sum over hours
  - Add ramp penalty term
  - Modify `total_emissions_objective()` to sum over hours
  - Handle representative day weighting

**2.5 Test Prototype**
- [ ] Create test script: `test_hourly_prototype.py`
  - Test 1-year model with 12 representative days
  - Verify ramp rate constraints are active (check dual variables)
  - Compare results to annual model
  - Expected model size:
    - Variables: 6 types × 12 days × 24 hours × 2 = 3,456 hourly variables
    - Constraints: ~5,000

**Files to create/modify:**
```
src/optimization/
├── model_hourly.py                            (NEW)
├── variables.py                               (UPDATE - add hourly variables)
├── constraints.py                             (UPDATE - activate ramp rates)
└── objectives.py                              (UPDATE - hourly sums)

tests/
└── test_hourly_prototype.py                   (NEW)
```

**Validation criteria:**
- ✓ Model solves in <5 minutes with HiGHS
- ✓ Ramp rate constraints are active (check constraint.active)
- ✓ Solution is feasible (all constraints satisfied)
- ✓ Hourly generation patterns make sense (nuclear baseload, hydro peaking)

---

### Phase 3: Full 20-Year Hourly Model (Week 3)
**Duration: 5-7 days**

#### Scope: 20 years, 12 representative days

**3.1 Extend Time Horizon**
- [ ] Update `model_hourly.py` to handle all 20 years
  - Extend hourly variables to all years
  - Maintain annual capacity planning (x[t,i], N[t,i])

**3.2 Performance Optimization**
- [ ] Set appropriate solver options
  - HiGHS: `presolve='on'`, `parallel='on'`
  - Consider Gurobi if available (free academic license)
  - Warm start: use annual model solution as initial guess

- [ ] Implement variable bounds tightening
  - Upper bound on `p_hourly[t,d,h,i]`: Use N[t,i] × capacity_factor[i]
  - Upper bound on `x[t,i]`: Estimate from demand growth

**3.3 Model Size Analysis**
- [ ] Calculate exact problem size
  ```
  Variables:
  - p_hourly: 20 years × 12 rep_days × 24 hours × 6 types = 34,560
  - ramp_violation: same = 34,560
  - x: 20 years × 6 types = 120
  - N: same = 120
  - TOTAL: 69,360 variables

  Constraints:
  - Demand (hourly): 20 × 12 × 24 = 5,760
  - Capacity (hourly): 20 × 12 × 24 × 6 = 34,560
  - Ramp up: same = 34,560
  - Ramp down: same = 34,560
  - Reserve margin (annual): 20
  - Capacity dynamics: 20 × 6 = 120
  - TOTAL: ~113,580 constraints
  ```

**3.4 Testing & Validation**
- [ ] Create comprehensive test suite: `test_hourly_full.py`
  - Test each year's generation profile
  - Verify ramp rate adherence
  - Check capacity factor utilization
  - Validate energy balance (hourly gen × day_weight = annual gen)

**3.5 Result Extraction**
- [ ] Update `model.py` `get_results()` method
  - Extract hourly generation profiles
  - Calculate hourly capacity factors
  - Identify peak hours and ramping events
  - Create hourly DataFrames

**Files to modify:**
```
src/optimization/
├── model_hourly.py                            (UPDATE - 20 years)
└── model.py                                   (UPDATE - get_results)

tests/
└── test_hourly_full.py                        (NEW)
```

**Validation criteria:**
- ✓ Model solves in <30 minutes with HiGHS (or <10 min with Gurobi)
- ✓ All ramp rate constraints satisfied
- ✓ Energy balance: hourly sums match annual demands
- ✓ Capacity factors realistic (nuclear ~90%, solar ~15%)

---

### Phase 4: Pareto Frontier with Hourly Model (Week 4)
**Duration: 3-5 days**

**4.1 Update Pareto Generation**
- [ ] Update `src/analysis/pareto.py`
  - Modify `weighted_sum_method()` to use hourly model
  - Pass `resolution='hourly'` to model constructor
  - Handle longer solve times (increase time_limit)

**4.2 Performance Considerations**
- [ ] Reduce Pareto points for hourly model
  - Annual model: 15 points
  - Hourly model: 7-10 points (computational cost ~10x higher)

- [ ] Implement parallel solving (optional)
  - Use multiprocessing to solve multiple α values simultaneously
  - Careful with solver licenses (Gurobi has token limits)

**4.3 Results Comparison**
- [ ] Create comparison script: `scripts/compare_annual_vs_hourly.py`
  - Run same scenario with annual and hourly models
  - Compare:
    - Total cost difference
    - Total emissions difference
    - Capacity mix differences
    - Computational time
  - Visualize differences

**Files to modify/create:**
```
src/analysis/
└── pareto.py                                  (UPDATE)

scripts/
└── compare_annual_vs_hourly.py                (NEW)
```

**Expected outcomes:**
- Hourly model will show **higher costs** (ramp constraints limit flexibility)
- More **hydro and gas** capacity (flexible ramping)
- Less **nuclear** capacity (inflexible baseload)
- Higher capacity factors for flexible plants

---

### Phase 5: Visualization & Analysis (Week 5)
**Duration: 3-5 days**

**5.1 New Visualizations for Hourly Model**
- [ ] Update `src/analysis/visualizations.py`

**New plot functions:**
```python
def plot_hourly_dispatch(results, year, rep_day_id):
    """
    Stacked area chart showing hourly generation by plant type
    for a representative day.

    X-axis: Hour (0-23)
    Y-axis: Power (MW)
    Stacks: Nuclear (bottom), Wind, Solar, Hydro, Gas, Biofuel (top)
    """

def plot_ramp_rates_heatmap(results, year):
    """
    Heatmap showing ramp rate utilization by plant type and hour.

    Color: Ramp rate as % of maximum (0% = blue, 100% = red)
    """

def plot_capacity_factor_hourly(results):
    """
    Box plot showing hourly capacity factor distribution by plant type.

    Shows min, 25%, median, 75%, max capacity factor across all hours.
    """

def plot_duck_curve(results, year):
    """
    Net load curve showing residual demand after renewables.

    Similar to California's "duck curve" visualization.
    """
```

**5.2 Analysis Reports**
- [ ] Create hourly analysis notebook: `notebooks/05_hourly_analysis.ipynb`
  - Compare annual vs hourly results
  - Analyze ramping patterns
  - Identify critical hours (peak demand, peak ramps)
  - Calculate ramping costs

**5.3 Documentation**
- [ ] Update README.md with hourly model instructions
- [ ] Create hourly model user guide: `docs/hourly_model_guide.md`
- [ ] Document representative day methodology

**Files to create/modify:**
```
src/analysis/
└── visualizations.py                          (UPDATE - add hourly plots)

notebooks/
└── 05_hourly_analysis.ipynb                   (NEW)

docs/
└── hourly_model_guide.md                      (NEW)

README.md                                      (UPDATE)
```

---

### Phase 6: Testing, Documentation & Integration (Week 6)
**Duration: 3-5 days**

**6.1 Comprehensive Testing**
- [ ] Create integration tests: `tests/test_hourly_integration.py`
  - Test annual → hourly data conversion
  - Test representative day clustering
  - Test ramp rate constraint logic
  - Test backward compatibility (annual model still works)

**6.2 Performance Benchmarking**
- [ ] Create benchmark script: `scripts/benchmark_models.py`
  - Time annual model vs hourly model
  - Test different numbers of representative days (4, 12, 24, 52)
  - Test different solvers (HiGHS, Gurobi)
  - Generate performance report

**6.3 Documentation Finalization**
- [ ] Complete all docstrings
- [ ] Create migration guide: `docs/migration_annual_to_hourly.md`
- [ ] Update CLAUDE.md with hourly model information

**6.4 Code Review & Cleanup**
- [ ] Remove debug print statements
- [ ] Ensure consistent naming conventions
- [ ] Add type hints to all functions
- [ ] Format code with Black

**Files to create:**
```
tests/
└── test_hourly_integration.py                 (NEW)

scripts/
└── benchmark_models.py                        (NEW)

docs/
├── hourly_model_guide.md                      (from Phase 5)
└── migration_annual_to_hourly.md              (NEW)

CLAUDE.md                                      (UPDATE)
```

---

## 5. Performance Optimization

### 5.1 Expected Model Sizes

| Configuration | Variables | Constraints | HiGHS Time | Gurobi Time |
|--------------|-----------|-------------|------------|-------------|
| **Annual (current)** | 360 | ~500 | <1 min | <10 sec |
| **1-year, 12 rep days** | 3,600 | ~5,000 | 2-5 min | 30-60 sec |
| **1-year, 24 rep days** | 7,200 | ~10,000 | 5-10 min | 1-2 min |
| **20-year, 12 rep days** | 69,360 | ~113,580 | 20-30 min | 5-10 min |
| **20-year, 24 rep days** | 138,720 | ~227,160 | 60+ min | 15-30 min |
| **20-year, full hourly** | 2.1M | ~3.5M | INFEASIBLE | 2-6 hours |

**Recommendation**: Use **12 representative days** for standard runs, **24 rep days** for final publication-quality results.

### 5.2 Solver Selection

**HiGHS (Open Source):**
- ✓ Free, no license needed
- ✓ Good for LP problems
- ✗ Slower on large MILP
- **Use for**: Development, testing, small-to-medium problems

**Gurobi (Commercial - Free Academic License):**
- ✓ Very fast on MILP
- ✓ Better presolve and cuts
- ✓ Parallel processing
- ✗ Requires license
- **Use for**: Large problems, Pareto frontier generation, final runs

**Solver options for hourly model:**
```python
# HiGHS
opt.options['presolve'] = 'on'
opt.options['parallel'] = 'on'
opt.options['time_limit'] = 1800  # 30 minutes
opt.options['mip_rel_gap'] = 0.02  # 2% gap acceptable

# Gurobi
opt.options['Presolve'] = 2        # Aggressive
opt.options['Method'] = 2          # Barrier method for LP
opt.options['Threads'] = 8         # Parallel threads
opt.options['TimeLimit'] = 600     # 10 minutes
opt.options['MIPGap'] = 0.01       # 1% gap
```

### 5.3 Representative Day Optimization

**Trade-off between accuracy and speed:**

| # Rep Days | Annual Energy Error | Peak Demand Error | Solve Time (20yr) |
|------------|--------------------|--------------------|-------------------|
| 4 | 5-8% | 10-15% | 5 min (Gurobi) |
| 12 | 2-3% | 3-5% | 10 min (Gurobi) |
| 24 | 1-2% | 2-3% | 25 min (Gurobi) |
| 52 | <1% | <2% | 60+ min (Gurobi) |
| 365 (full) | 0% | 0% | HOURS |

**Recommended clustering strategies:**

**Strategy A: Seasonal (12 clusters)**
- Winter weekday peak (Dec-Feb)
- Winter weekend (Dec-Feb)
- Spring shoulder weekday (Mar-May)
- Spring shoulder weekend (Mar-May)
- Summer weekday peak (Jun-Aug)
- Summer weekend (Jun-Aug)
- Fall shoulder weekday (Sep-Nov)
- Fall shoulder weekend (Sep-Nov)
- Extreme peak day
- Extreme low day
- Renewable generation high day
- Renewable generation low day

**Strategy B: High-fidelity (24 clusters)**
- 4 seasons × 3 day types × 2 demand levels

### 5.4 Warm Starting

Use annual model solution to initialize hourly model:

```python
def warm_start_hourly_from_annual(hourly_model, annual_results):
    """
    Initialize hourly model with annual model solution.

    Capacity variables (x, N): Copy directly
    Hourly generation (p_hourly): Distribute annual p[t,i] uniformly
    """
    m = hourly_model.model

    # Set capacity variables
    for t in m.years:
        for i in m.plant_types:
            m.x[t, i].set_value(annual_results['new_builds'].loc[(t,i), 'new_capacity_MW'])
            m.N[t, i].set_value(annual_results['capacity'].loc[(t,i), 'total_capacity_MW'])

    # Distribute annual generation uniformly across hours
    for t in m.years:
        for i in m.plant_types:
            annual_gen_mwh = annual_results['generation'].loc[(t,i), 'generation_MWh']
            hourly_gen_mw = annual_gen_mwh / 8760  # Average MW

            for d in m.rep_days:
                for h in m.hours_per_day:
                    # Could use load profile to make this smarter
                    m.p_hourly[t, d, h, i].set_value(hourly_gen_mw)
```

---

## 6. Testing Strategy

### 6.1 Unit Tests

**File: `tests/test_hourly_model.py`**

```python
def test_ramp_rate_constraint_active():
    """Verify ramp rate constraints are present and active."""
    model = PowerSystemOptimizationHourly(
        start_year=2025, end_year=2025,
        use_rep_days=True, n_rep_days=12
    )
    model.build_model()

    # Check constraint exists
    assert hasattr(model.model, 'ramp_up_constraint')
    assert hasattr(model.model, 'ramp_down_constraint')

    # Check constraints are not skipped
    num_ramp_constraints = sum(
        1 for c in model.model.ramp_up_constraint.values()
    )
    assert num_ramp_constraints > 0

def test_hourly_energy_balance():
    """Verify hourly generation sums to annual demand."""
    model = PowerSystemOptimizationHourly(...)
    results = model.optimize()

    # Sum hourly generation across all hours and plants
    hourly_total = results['hourly_generation'].groupby('year')['generation_MW'].sum()

    # Compare to annual demand (allowing for representative day weighting)
    annual_demand = model.demand_data['annual_demand'] * 1000  # GWh to MWh

    # Should match within 1%
    assert np.allclose(hourly_total, annual_demand, rtol=0.01)

def test_capacity_factor_limits():
    """Verify no plant exceeds its capacity factor."""
    model = PowerSystemOptimizationHourly(...)
    results = model.optimize()

    for plant_type in model.plant_params['capex'].keys():
        max_cf = model.plant_params['capacity_factor'][plant_type]
        actual_cf = calculate_capacity_factor(results, plant_type)

        assert actual_cf <= max_cf + 0.01  # Allow 1% tolerance

def test_ramp_rate_adherence():
    """Verify all ramp rates are within limits."""
    model = PowerSystemOptimizationHourly(...)
    results = model.optimize()

    hourly_gen = results['hourly_generation']

    for plant_type in model.plant_params['capex'].keys():
        plant_data = hourly_gen[hourly_gen['plant_type'] == plant_type]

        # Calculate hour-to-hour changes
        ramp_rates = plant_data['generation_MW'].diff()
        max_allowed_ramp = (model.plant_params['ramp_rate'][plant_type]
                           * plant_data['capacity_MW'] * 60)

        # Check violations (allowing for soft constraint violations if enabled)
        violations = ramp_rates.abs() > max_allowed_ramp
        assert violations.sum() < 0.05 * len(ramp_rates)  # <5% violations OK
```

### 6.2 Integration Tests

**File: `tests/test_hourly_integration.py`**

```python
def test_annual_vs_hourly_consistency():
    """Compare annual and hourly model results."""
    # Run both models with same parameters
    annual_model = PowerSystemOptimization(start_year=2025, end_year=2030)
    annual_results = annual_model.optimize()

    hourly_model = PowerSystemOptimizationHourly(start_year=2025, end_year=2030)
    hourly_results = hourly_model.optimize()

    # Costs should be similar (hourly may be slightly higher due to ramp constraints)
    cost_diff = (hourly_results['total_cost'] - annual_results['total_cost']) / annual_results['total_cost']
    assert cost_diff < 0.10  # Hourly costs at most 10% higher

    # Emissions should be similar
    emis_diff = abs(hourly_results['total_emissions'] - annual_results['total_emissions']) / annual_results['total_emissions']
    assert emis_diff < 0.05  # Within 5%

def test_representative_days_coverage():
    """Verify representative days cover all seasons."""
    rep_days = load_representative_days(n_clusters=12)

    # Should have days from all 4 seasons
    seasons = [get_season(day['source_day']) for day in rep_days]
    assert len(set(seasons)) == 4  # All 4 seasons represented

    # Weights should sum to 365
    total_weight = sum(day['weight'] for day in rep_days)
    assert total_weight == 365
```

### 6.3 Performance Tests

**File: `tests/test_performance.py`**

```python
def test_solve_time_acceptable():
    """Ensure model solves in reasonable time."""
    import time

    model = PowerSystemOptimizationHourly(
        start_year=2025, end_year=2045,
        use_rep_days=True, n_rep_days=12
    )

    start = time.time()
    model.optimize(solver='highs', time_limit=1800)
    elapsed = time.time() - start

    # Should solve in under 30 minutes with HiGHS
    assert elapsed < 1800, f"Solve time {elapsed:.0f}s exceeds 30min limit"
```

### 6.4 Validation Tests

```python
def test_physical_realism():
    """Verify results pass basic sanity checks."""
    model = PowerSystemOptimizationHourly(...)
    results = model.optimize()

    # 1. Nuclear should have highest capacity factor
    cf = calculate_capacity_factors(results)
    assert cf['nuclear'] > cf['wind']
    assert cf['nuclear'] > cf['solar']

    # 2. Hydro should show more ramping than nuclear
    ramp_variance = calculate_ramp_variance(results)
    assert ramp_variance['hydro'] > ramp_variance['nuclear']

    # 3. Solar generation should peak mid-day
    solar_profile = get_daily_profile(results, 'solar')
    peak_hour = solar_profile.argmax()
    assert 10 <= peak_hour <= 14  # Peak between 10am-2pm

    # 4. Wind should have lower correlation with demand
    wind_gen = results['hourly_generation'][results['hourly_generation']['plant_type']=='wind']
    demand = results['hourly_demand']
    correlation = wind_gen['generation_MW'].corr(demand)
    assert correlation < 0.5  # Wind not well-correlated with demand
```

---

## 7. Risks and Mitigation

### 7.1 Computational Complexity

**Risk**: Full hourly model (20 years × 8,760 hours) is computationally intractable.

**Mitigation**:
- ✓ **Use representative days** (12-24 clusters) - reduces problem size by 97%
- ✓ **Use Gurobi** instead of HiGHS for large problems (10x faster)
- ✓ **Warm start** from annual model solution
- ✓ **Parallel solving** for Pareto frontier generation
- ✓ **Relax MIP gap** to 1-2% (vs 0.1% for annual model)

**Contingency**: If still too slow, reduce to:
- **5-year horizon** instead of 20 years (test case)
- **4-6 representative days** instead of 12
- **Soft ramp constraints** with penalties (easier to solve than hard constraints)

### 7.2 Data Availability

**Risk**: Real hourly demand data for Ontario may not be publicly available or may require payment.

**Mitigation**:
- ✓ **IESO provides free data** - https://www.ieso.ca/power-data (confirmed available)
- ✓ **Synthetic demand generator** already planned as fallback
- ✓ **Use 2023-2024 data** as template for future years (scale to match annual forecasts)

**Contingency**: If IESO data unavailable:
- Use **ERCOT (Texas) or CAISO (California)** hourly data as template
- **Scale and shift** to match Ontario characteristics (higher winter peak)

### 7.3 Model Convergence

**Risk**: Hourly model may be infeasible or fail to converge due to tight ramp constraints.

**Mitigation**:
- ✓ **Use soft ramp constraints** with penalty (recommended approach)
- ✓ **Relax initial capacity constraints** (allow more flexible plants initially)
- ✓ **Increase reserve margin** to 20% (vs 15%) to ensure sufficient flexible capacity
- ✓ **Debugging mode**: Start with no ramp constraints, gradually tighten

**Contingency**:
- **Identify binding constraints** using dual variables
- **Add flexible capacity** manually if model is infeasible (e.g., +1 GW hydro)
- **Relax ramp rates** by 20% as sensitivity check

### 7.4 Memory Constraints

**Risk**: Large models may exceed available RAM (especially on laptops).

**Mitigation**:
- ✓ **Representative days** keep model under 100k variables (fits in 8 GB RAM)
- ✓ **Use sparse matrix solvers** (HiGHS and Gurobi both support this)
- ✓ **Don't store full hourly results** - aggregate to daily or weekly for visualization

**Monitoring**:
```python
import psutil
print(f"Memory usage: {psutil.virtual_memory().percent}%")
```

**Contingency**: If memory issues persist:
- Run on **cloud compute** (AWS, Google Colab with high RAM)
- Use **Google Colab Pro** (25-52 GB RAM available)

### 7.5 Backward Compatibility

**Risk**: Hourly model changes break existing annual model.

**Mitigation**:
- ✓ **Keep annual model intact** - create separate `model_hourly.py`
- ✓ **Use resolution parameter** to switch between modes
- ✓ **Maintain same API** - `optimize()`, `get_results()` work for both
- ✓ **Comprehensive testing** of both models

**Testing**:
```python
# Ensure annual model still works
def test_annual_model_unchanged():
    model = PowerSystemOptimization(start_year=2025, end_year=2045)
    results = model.optimize()
    assert results is not None
    assert 'total_cost' in results
```

### 7.6 Results Interpretation

**Risk**: Hourly results are harder to interpret and visualize than annual results.

**Mitigation**:
- ✓ **New visualization tools** (Phase 5) - hourly dispatch charts, ramp heatmaps
- ✓ **Aggregated summaries** - daily, weekly, monthly averages
- ✓ **Interactive dashboards** (optional Plotly Dash app)
- ✓ **Clear documentation** in notebooks

**Key metrics to track**:
- Average hourly capacity factor by plant type
- Number of ramping events per year
- Hours of peak demand vs peak generation
- Renewable curtailment hours (if any)

---

## 8. File Structure Summary

```
Ontario-Power-Planning/
├── data/
│   ├── raw/
│   │   ├── Ontario_Energy_Data_Summary.md          (existing)
│   │   └── ieso_hourly_demand_2023_2024.csv        (NEW - Phase 1)
│   ├── processed/
│   │   ├── plant_parameters.json                   (existing - has ramp_rate)
│   │   ├── demand_forecast.csv                     (existing)
│   │   ├── initial_capacity.json                   (existing)
│   │   ├── hourly_demand_2025_2045.csv             (NEW - Phase 1)
│   │   └── representative_days_12clusters.json     (NEW - Phase 1)
│   ├── load_data.py                                (UPDATE - Phase 1)
│   ├── create_synthetic_demand.py                  (NEW - Phase 1)
│   └── create_representative_days.py               (NEW - Phase 1)
│
├── src/
│   ├── optimization/
│   │   ├── model.py                                (existing - backward compatible)
│   │   ├── model_hourly.py                         (NEW - Phase 2)
│   │   ├── variables.py                            (UPDATE - Phase 2)
│   │   ├── constraints.py                          (UPDATE - Phase 2 - ACTIVATE ramp!)
│   │   └── objectives.py                           (UPDATE - Phase 2)
│   ├── analysis/
│   │   ├── pareto.py                               (UPDATE - Phase 4)
│   │   └── visualizations.py                       (UPDATE - Phase 5)
│   └── utils/
│       └── time_series.py                          (NEW - Phase 1)
│
├── tests/
│   ├── test_hourly_prototype.py                    (NEW - Phase 2)
│   ├── test_hourly_full.py                         (NEW - Phase 3)
│   ├── test_hourly_integration.py                  (NEW - Phase 6)
│   └── test_performance.py                         (NEW - Phase 6)
│
├── notebooks/
│   └── 05_hourly_analysis.ipynb                    (NEW - Phase 5)
│
├── scripts/
│   ├── compare_annual_vs_hourly.py                 (NEW - Phase 4)
│   └── benchmark_models.py                         (NEW - Phase 6)
│
├── docs/
│   ├── hourly_model_guide.md                       (NEW - Phase 5)
│   └── migration_annual_to_hourly.md               (NEW - Phase 6)
│
├── hourly_power_integration.md                     (THIS FILE)
├── CLAUDE.md                                       (UPDATE - Phase 6)
└── README.md                                       (UPDATE - Phase 5)
```

---

## 9. Success Criteria

### Phase 1 (Data)
- ✓ Hourly demand data available for all years (2025-2045)
- ✓ Representative days cluster with R² > 0.95 vs full profile
- ✓ Peak demand preserved (within 5% of annual peak)

### Phase 2 (Prototype)
- ✓ 1-year hourly model solves in <5 minutes
- ✓ Ramp rate constraints are active (check `.active` property)
- ✓ All constraints satisfied (feasible solution)

### Phase 3 (Full Model)
- ✓ 20-year hourly model solves in <30 minutes with Gurobi
- ✓ Energy balance: hourly gen sums match annual demands
- ✓ Ramp rates within limits (or violations < 5% if soft constraints)

### Phase 4 (Pareto)
- ✓ Generate 7-10 Pareto points with hourly model
- ✓ Results show sensible differences from annual model (more flexible capacity)
- ✓ Cost-emissions trade-offs preserved

### Phase 5 (Visualization)
- ✓ Publication-quality plots of hourly dispatch
- ✓ Ramp rate analysis showing plant utilization
- ✓ Clear documentation and examples

### Phase 6 (Production)
- ✓ All tests pass (unit, integration, performance)
- ✓ Annual model still works (backward compatible)
- ✓ Complete documentation for users

---

## 10. Timeline Summary

| Week | Phase | Key Deliverables | Estimated Hours |
|------|-------|------------------|-----------------|
| 1 | Data Preparation | Hourly demand data, rep days clustering | 20-30 hours |
| 2 | Prototype Hourly | 1-year model with active ramp constraints | 30-40 hours |
| 3 | Full Model | 20-year hourly optimization | 30-40 hours |
| 4 | Pareto Frontier | Multi-objective with hourly resolution | 20-30 hours |
| 5 | Visualization | Hourly dispatch plots, analysis tools | 20-30 hours |
| 6 | Testing & Docs | Complete test suite, user guide | 20-30 hours |

**Total Duration: 6 weeks (140-200 hours)**

**Critical Path**: Data → Prototype → Full Model → Pareto → Docs

**Parallel Tracks**:
- Weeks 1-2: Can develop data loading and prototype simultaneously
- Weeks 5-6: Visualization and testing can overlap

---

## 11. Next Steps - Getting Started

### Immediate Actions (Start Today):

**Option A: Download Real IESO Data (2 hours)**
1. Visit https://www.ieso.ca/power-data
2. Navigate to historical demand data
3. Download 2023-2024 hourly Ontario demand (CSV)
4. Save to `data/raw/ieso_hourly_demand_2023_2024.csv`
5. Validate: Check for missing hours, data quality

**Option B: Build Synthetic Demand Generator (4 hours)**
1. Create `data/create_synthetic_demand.py`
2. Implement seasonal, daily, weekly patterns
3. Scale to match existing annual forecasts
4. Validate: Plot 8760-hour profile, check totals

**Next Actions (This Week - Phase 1):**

1. **Representative Day Clustering** (6 hours)
   - Create `data/create_representative_days.py`
   - Implement K-means clustering (k=12)
   - Validate clustering quality (R², energy balance)
   - Generate `data/processed/representative_days_12clusters.json`

2. **Data Loading Utilities** (2 hours)
   - Update `data/load_data.py`
   - Add `load_hourly_demand()` function
   - Add `load_representative_days()` function
   - Write unit tests for data loading

3. **Visualization & Validation** (2 hours)
   - Plot all 365 days vs 12 representative days
   - Compare peaks, valleys, seasonal patterns
   - Validate weights sum to 365
   - Document clustering methodology

**After Week 1:**
- Review data quality and clustering results
- Decide: Proceed to Phase 2 (Prototype) or refine clustering
- Checkpoint: Do representative days accurately capture demand patterns?

---

## 12. Technical References

### Representative Days Clustering
- Nahmmacher et al. (2016) "Carpe diem: A novel approach to select representative days for long-term power system modeling"
- Pineda & Morales (2018) "Chronological time-period clustering for optimal capacity expansion planning with storage"

### Ramp Rate Modeling
- Palmintier & Webster (2011) "Impact of unit commitment constraints on generation expansion planning with renewables"
- Shortt et al. (2013) "Incorporating flexibility requirements into generation expansion planning"

### Ontario Energy System
- IESO Data Portal: https://www.ieso.ca/power-data
- IESO Planning Outlook: https://www.ieso.ca/en/Sector-Participants/Planning-and-Forecasting/Annual-Planning-Outlook
- Ontario Energy Report 2023: https://www.ieso.ca/corporate-ieso/media/reports

### Code Examples & Open Source Models
- NREL ReEDS (Regional Energy Deployment System): https://github.com/NREL/ReEDS-2.0
- Calliope Energy System Model: https://github.com/calliope-project/calliope
- PyPSA (Python for Power System Analysis): https://github.com/PyPSA/PyPSA

---

## 13. Key Design Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Time resolution** | Representative days (12-24) | 97% size reduction, <3% error |
| **Ramp constraints** | Soft with $1000/MW penalty | Easier to solve, more realistic than hard constraints |
| **Primary solver** | Gurobi (HiGHS for dev) | 10× faster, handles 70k+ variables efficiently |
| **Model architecture** | Separate `model_hourly.py` | Maintains backward compatibility with annual model |
| **Data source** | IESO + synthetic fallback | Real data preferred, synthetic ensures project continuity |
| **Clustering algorithm** | K-means on demand features | Industry standard, well-validated approach |
| **Variable naming** | `p_hourly[t,d,h,i]` | Clear distinction from annual `p[t,i]` |
| **Energy balance** | Representative day weights | Ensures hourly totals match annual demands |

---

## 14. Expected Results & Insights

### Quantitative Differences (Annual vs Hourly Model)

**Cost Impact:**
- Annual model: $125B (cost-optimal)
- Hourly model: $138-145B (estimated 10-15% higher)
- **Reason**: Ramp constraints require more flexible (expensive) capacity

**Capacity Mix Changes:**
- **Nuclear**: Decrease 10-20% (inflexible baseload less valuable)
- **Hydro**: Increase 30-50% (fast ramping highly valued)
- **Gas**: Increase 20-30% (flexible peaking)
- **Wind/Solar**: Similar (weather-dependent, not for flexibility)

**Operational Insights:**
- Nuclear operates at 85-90% capacity factor (baseload)
- Hydro shows high variability: 20-80% CF (load following)
- Gas turbines peak during ramps (morning/evening)
- Solar/wind capacity factors unchanged (weather-driven)

**Ramping Analysis:**
- Hydro provides 60-70% of ramping services
- Gas provides 20-30% of ramping services
- Nuclear contributes <5% to ramping (too slow)
- Peak ramps occur: 6-9am (morning) and 5-8pm (evening)

### Qualitative Insights

**Policy Implications:**
1. **Flexibility has value** - Fast-ramping plants justify higher capital costs
2. **Nuclear limitations** - Baseload nuclear less attractive in renewable-heavy grid
3. **Hydro is critical** - Existing hydro capacity is key enabler of renewable integration
4. **Storage opportunities** - Large ramps suggest battery storage could be valuable

**Model Realism:**
- Hourly model captures actual grid operations
- Annual model optimistic about nuclear economics
- Ramp constraints bind during transitions (dawn/dusk)
- Reserve margin may need to increase for reliability

---

## 15. Conclusion

This implementation plan provides a **structured, phased approach** to integrating hourly resolution into the Ontario Power Plant Optimization model. The key innovation is **representative day clustering**, which achieves a **97% reduction in computational complexity** while maintaining **95%+ accuracy**.

### Key Benefits of Hourly Model:
1. ✓ **Realistic ramp constraints** - properly values flexible generation (hydro, gas)
2. ✓ **Operational insights** - shows when plants ramp, not just annual totals
3. ✓ **Better renewable integration** - models solar/wind intermittency
4. ✓ **Policy relevance** - aligns with grid operator (IESO) operational planning
5. ✓ **Technology comparison** - fairly evaluates baseload vs flexible capacity

### Risks Are Manageable:
- Representative days solve computational complexity
- Soft ramp constraints prevent infeasibility
- Backward compatibility preserves annual model
- Synthetic demand ensures data availability

### Timeline Is Achievable:
- **6 weeks** for complete implementation
- **Incremental validation** at each phase
- **Parallel development** where possible

### Ready to Begin:
The plan is production-ready and can start immediately with **Phase 1 (Data Preparation)**. The first milestone is obtaining hourly demand data and creating representative day clusters.

---

**Project Status**: Ready for implementation
**Next Step**: Download IESO hourly data or create synthetic demand generator
**Critical Success Factor**: High-quality representative day clustering (R² > 0.95)
**Expected Completion**: 6 weeks from start date