# Mathematical Formulation: Annual Resolution Model (Current Implementation)
## Ontario Power Plant Optimization

---

## Model Overview

**Type**: Multi-objective Linear Program (LP)
**Time horizon**: 20 years (2025-2045)
**Temporal resolution**: Annual (single energy value per year)
**Plant types**: Nuclear, Wind, Solar, Natural Gas, Hydro, Biofuel
**Purpose**: Determine optimal power plant capacity expansion to minimize cost and emissions while meeting demand
**Status**: ✅ Fully implemented and validated (Phases 1-4 complete)

---

## Sets and Indices

| Symbol | Description | Size |
|--------|-------------|------|
| $t \in T$ | Years in planning horizon | $\|T\| = 20$ (2025-2045) |
| $i \in I$ | Plant types | $\|I\| = 6$ (nuclear, wind, solar, gas, hydro, biofuel) |

---

## Decision Variables

| Variable | Domain | Units | Description |
|----------|--------|-------|-------------|
| $x_{t,i}$ | $\mathbb{R}_+$ | MW | New capacity of plant type $i$ to build in year $t$ |
| $p_{t,i}$ | $\mathbb{R}_+$ | MWh | Annual energy generation from plant type $i$ in year $t$ |
| $N_{t,i}$ | $\mathbb{R}_+$ | MW | Total operating capacity of plant type $i$ in year $t$ |

**Total variables**: $3 \times |T| \times |I| = 3 \times 20 \times 6 = 360$

**Variable relationships**:
- $x_{t,i}$: Capacity decision (how much to build each year)
- $N_{t,i}$: State variable (total capacity available)
- $p_{t,i}$: Dispatch decision (how much energy to generate annually)

---

## Parameters

### Plant Technology Parameters

**File**: `data/processed/plant_parameters.json`

| Parameter | Units | Description | Example Values |
|-----------|-------|-------------|----------------|
| $CapEx_i$ | \$/kW | Capital cost per kW capacity | Nuclear: $17,500, Gas: $1,500 |
| $OpEx_i$ | \$/MWh | Operating cost per MWh generated | Nuclear: $22, Gas: $55 |
| $MainEx_i$ | \$/MW/year | Annual maintenance cost per MW capacity | Nuclear: $105,000, Wind: $45,000 |
| $EmissionFactor_i$ | tons CO₂/MWh | Carbon emissions per MWh generated | Gas: 0.45, Nuclear: 0.012 |
| $CF_i$ | - | Capacity factor (0-1) | Nuclear: 0.90, Solar: 0.15 |
| $LeadTime_i$ | years | Construction lead time | Nuclear: 7, Wind: 2 |
| $Lifespan_i$ | years | Plant operational lifetime | Nuclear: 60, Wind: 25 |
| $RampRate_i$ | MW/min per MW | Maximum ramp rate (NOT USED in annual model) | Hydro: 0.15, Nuclear: 0.02 |

### System Parameters

**File**: `src/optimization/model.py` (lines 30-31, 178)

| Parameter | Value | Description |
|-----------|-------|-------------|
| $r$ | 3.92% | Real discount rate (Canada long-term rate) |
| $RM$ | 15% | Reserve margin above peak demand |

### Time-Varying Parameters

**File**: `data/processed/demand_forecast.csv`, `data/processed/initial_capacity.json`

| Parameter | Units | Description |
|-----------|-------|-------------|
| $AnnualDemand_t$ | MWh | Total electricity demand in year $t$ |
| $PeakDemand_t$ | MW | Peak hourly demand in year $t$ |
| $InitialCapacity_i$ | MW | Existing capacity of plant type $i$ in year $t_0 = 2025$ |
| $Retirement_{t,i}$ | MW | Capacity retiring in year $t$ for plant type $i$ |
| $\delta_\tau$ | - | Discount factor: $\delta_\tau = \frac{1}{(1+r)^\tau}$ where $\tau = t - t_0$ |

**Demand growth**: 2.2% annually (151 TWh in 2025 → 193 TWh in 2045)

---

## Objective Functions

### 1. Total System Cost (Minimize)

$$
\min Z_{cost} = \sum_{\tau=0}^{|T|-1} \delta_\tau \left[ \underbrace{\sum_{i \in I} x_{t_0+\tau,i} \cdot CapEx_i \cdot 1000}_{\text{Capital Costs}} + \underbrace{\sum_{i \in I} p_{t_0+\tau,i} \cdot OpEx_i}_{\text{Operating Costs}} + \underbrace{\sum_{i \in I} N_{t_0+\tau,i} \cdot MainEx_i}_{\text{Maintenance Costs}} \right]
$$

**Components**:

1. **Capital Costs (CAPEX)**: Discounted NPV of new capacity investments
   $$CapitalCost = \sum_{\tau=0}^{19} \frac{1}{(1+r)^\tau} \sum_{i \in I} x_{t_0+\tau,i} \cdot CapEx_i \cdot 1000$$
   - $CapEx_i$ is in \$/kW, multiply by 1000 to convert to \$/MW
   - Discounted to present value using $\delta_\tau = \frac{1}{(1.0392)^\tau}$

2. **Operating Costs (OPEX)**: Discounted cost of generation
   $$OperatingCost = \sum_{\tau=0}^{19} \frac{1}{(1+r)^\tau} \sum_{i \in I} p_{t_0+\tau,i} \cdot OpEx_i$$
   - $p_{t,i}$ is annual energy (MWh), $OpEx_i$ is cost per MWh
   - Variable cost that depends on how much is generated

3. **Maintenance Costs**: Discounted annual maintenance of operating capacity
   $$MaintenanceCost = \sum_{\tau=0}^{19} \frac{1}{(1+r)^\tau} \sum_{i \in I} N_{t_0+\tau,i} \cdot MainEx_i$$
   - Fixed annual cost per MW of installed capacity
   - Independent of generation (must pay even if plant is idle)

**Implementation**: `src/optimization/objectives.py` (lines 13-53)

**Discount factor**: $\delta_\tau = \frac{1}{(1+r)^\tau}$ where $\tau = t - 2025$ (years from start)
- Year 0 (2025): $\delta_0 = 1.000$
- Year 10 (2035): $\delta_{10} = 0.683$
- Year 20 (2045): $\delta_{20} = 0.466$

### 2. Total Carbon Emissions (Minimize)

$$
\min Z_{emissions} = \sum_{t \in T} \sum_{i \in I} p_{t,i} \cdot EmissionFactor_i
$$

**Description**: Total lifecycle carbon emissions over planning horizon (tons CO₂)

**Implementation**: `src/optimization/objectives.py` (lines 56-74)

**Example emission factors**:
- Gas: 0.45 tons CO₂/MWh (highest emitter)
- Nuclear: 0.012 tons CO₂/MWh (lifecycle emissions from construction/mining)
- Wind: 0.011 tons CO₂/MWh (lowest)

### 3. Multi-Objective Weighted Sum

$$
\min Z_{multi} = \alpha \cdot \frac{Z_{cost}}{Z_{cost}^{max}} + (1-\alpha) \cdot \frac{Z_{emissions}}{Z_{emissions}^{max}}
$$

where:
- $\alpha \in [0, 1]$: Weight parameter controlling trade-off
  - $\alpha = 1.0$: Pure cost minimization (cheapest solution, likely high emissions)
  - $\alpha = 0.0$: Pure emissions minimization (cleanest solution, likely expensive)
  - $0 < \alpha < 1$: Trade-off between cost and emissions
- $Z_{cost}^{max}$: Normalization factor (cost from emissions-optimal solution)
- $Z_{emissions}^{max}$: Normalization factor (emissions from cost-optimal solution)

**Implementation**: `src/optimization/objectives.py` (lines 77-104)

**Pareto frontier generation**: Solve for $\alpha \in \{0.0, 0.1, 0.2, ..., 1.0\}$ to trace cost-emissions trade-off curve

---

## Constraints

### 1. Annual Demand Satisfaction

$$
\sum_{i \in I} p_{t,i} \geq AnnualDemand_t \quad \forall t \in T
$$

**Description**: Total annual generation must meet or exceed annual demand.

**Units**: MWh (energy, not power)

**Implementation**: `src/optimization/constraints.py` (lines 15-27)

**Number of constraints**: $|T| = 20$

**Example for 2025**:
- $AnnualDemand_{2025} = 151,000$ GWh = 151,000,000 MWh
- $\sum_i p_{2025,i} \geq 151,000,000$ MWh

---

### 2. Annual Capacity Constraint

$$
p_{t,i} \leq N_{t,i} \cdot CF_i \cdot 8760 \quad \forall t \in T, \forall i \in I
$$

**Description**: Annual generation cannot exceed available capacity times capacity factor times hours per year.

**Derivation**:
- $N_{t,i}$ = Installed capacity (MW)
- $CF_i$ = Capacity factor (fraction of time at full output)
- $8760$ = Hours per year
- Maximum annual energy = $N_{t,i} \times CF_i \times 8760$ MWh

**Implementation**: `src/optimization/constraints.py` (lines 30-45)

**Number of constraints**: $|T| \times |I| = 20 \times 6 = 120$

**Physical interpretation**:
- Nuclear ($CF = 0.90$): 1000 MW plant → max 7,884,000 MWh/year (90% of full capacity)
- Solar ($CF = 0.15$): 1000 MW plant → max 1,314,000 MWh/year (15% of full capacity)
- Capacity factor accounts for:
  - Forced outages
  - Maintenance downtime
  - Resource availability (solar only during day, wind when windy)

**Example for nuclear in 2025**:
- $N_{2025,nuclear} = 13,000$ MW (initial capacity)
- $p_{2025,nuclear} \leq 13,000 \times 0.90 \times 8760 = 102,492,000$ MWh

---

### 3. Reserve Margin Constraint

$$
\sum_{i \in I} N_{t,i} \geq (1 + RM) \cdot PeakDemand_t \quad \forall t \in T
$$

**Description**: Total installed capacity must exceed peak demand by reserve margin (15%).

**Purpose**: Ensures system reliability by maintaining excess capacity for:
- Forced outages (unexpected plant failures)
- Scheduled maintenance
- Demand forecast uncertainty
- Transmission constraints

**Implementation**: `src/optimization/constraints.py` (lines 48-62)

**Number of constraints**: $|T| = 20$

**Example for 2025**:
- $PeakDemand_{2025} = 24,000$ MW
- $RM = 0.15$ (15%)
- Required capacity: $(1 + 0.15) \times 24,000 = 27,600$ MW
- Current capacity: $\sum_i InitialCapacity_i = 40,449$ MW ✓ (exceeds requirement)

**Note**: Reserve margin is based on **peak demand** (MW), not annual demand (MWh)

---

### 4. Ramp Rate Constraint (DISABLED)

$$
\text{SKIPPED - Not applicable at annual resolution}
$$

**Status**: Disabled in current implementation

**Reason**: Annual model has no sub-annual time resolution, so hour-to-hour ramp rates cannot be modeled.

**Implementation**: `src/optimization/constraints.py` (lines 65-78) - empty function

**Future work**: Will be activated in hourly resolution model (Phase 7)

**Mathematical form** (for future reference):
$$
|p_{t,i,h} - p_{t,i,h-1}| \leq RampRate_i \times N_{t,i} \times 60 \quad \forall t, i, h
$$
where $h$ indexes hours within the year.

---

### 5. Capacity Dynamics with Lead Times and Retirements

#### 5a. Initial Capacity Constraint

$$
N_{t_0,i} = InitialCapacity_i \quad \forall i \in I
$$

**Description**: Fix capacity in first year (2025) to existing capacity.

**Implementation**: `src/optimization/constraints.py` (lines 97-99)

**Number of constraints**: $|I| = 6$

**Initial capacity (2025)**:

| Plant Type | Capacity (MW) | Share |
|------------|---------------|-------|
| Nuclear | 13,000 | 32.1% |
| Gas | 10,500 | 26.0% |
| Hydro | 8,500 | 21.0% |
| Wind | 5,575 | 13.8% |
| Solar | 2,669 | 6.6% |
| Biofuel | 205 | 0.5% |
| **Total** | **40,449** | **100%** |

#### 5b. Capacity Evolution Constraint

$$
N_{t,i} = N_{t-1,i} + x_{t-LeadTime_i,i} - Retirement_{t,i} \quad \forall t \in T \setminus \{t_0\}, \forall i \in I
$$

where:
$$
x_{t-LeadTime_i,i} =
\begin{cases}
x_{t-LeadTime_i,i} & \text{if } t - LeadTime_i \geq t_0 \text{ and } t - LeadTime_i \in T \\
0 & \text{otherwise}
\end{cases}
$$

**Description**: Tracks capacity evolution accounting for:
1. **Previous capacity**: $N_{t-1,i}$ (carry forward)
2. **New capacity coming online**: $x_{t-LeadTime_i,i}$ (plants built $LeadTime_i$ years ago)
3. **Retirements**: $Retirement_{t,i}$ (plants reaching end of life)

**Implementation**: `src/optimization/constraints.py` (lines 101-136)

**Number of constraints**: $(|T|-1) \times |I| = 19 \times 6 = 114$

**Lead times by plant type**:

| Plant Type | Lead Time (years) | Example |
|------------|-------------------|---------|
| Nuclear | 7 | Build in 2025 → online in 2032 |
| Hydro | 6 | Build in 2025 → online in 2031 |
| Gas | 3 | Build in 2025 → online in 2028 |
| Biofuel | 3 | Build in 2025 → online in 2028 |
| Wind | 2 | Build in 2025 → online in 2027 |
| Solar | 2 | Build in 2025 → online in 2027 |

**Example for nuclear in 2032**:
$$N_{2032,nuclear} = N_{2031,nuclear} + x_{2025,nuclear} - Retirement_{2032,nuclear}$$
- New capacity built in 2025 becomes operational 7 years later in 2032

**Retirement model** (Phase 3 enhancement):
- Assumes initial capacity was built uniformly over past lifespan
- Annual retirement rate: $Retirement_{t,i} = \frac{InitialCapacity_i}{Lifespan_i}$
- Continues for first $Lifespan_i$ years of planning horizon

**Example retirements**:
- Nuclear (lifespan 60 years): $\frac{13,000}{60} = 217$ MW/year
- Wind (lifespan 25 years): $\frac{5,575}{25} = 223$ MW/year
- Total retirements: ~920 MW/year → 18.4 GW over 20 years

**Impact**: Retirements dominate capacity needs
- Without retirements: 4.36 GW new capacity needed (demand growth only)
- With retirements: 23.75 GW new capacity needed (5.4× more!)

---

### 6. Non-Negativity Constraints

$$
\begin{align}
x_{t,i} &\geq 0 \quad \forall t \in T, \forall i \in I \\
N_{t,i} &\geq 0 \quad \forall t \in T, \forall i \in I \\
p_{t,i} &\geq 0 \quad \forall t \in T, \forall i \in I
\end{align}
$$

**Description**: All decision variables must be non-negative.

**Interpretation**:
- Cannot build negative capacity
- Cannot have negative total capacity
- Cannot generate negative energy

**Implementation**: Enforced by variable domain specification in `src/optimization/variables.py` (lines 26-47)
```python
domain=pyo.NonNegativeReals
```

---

## Problem Size Summary

### Current Annual Model

| Component | Count | Calculation |
|-----------|-------|-------------|
| **Variables** | | |
| New builds ($x_{t,i}$) | 120 | $20 \times 6$ |
| Total capacity ($N_{t,i}$) | 120 | $20 \times 6$ |
| Generation ($p_{t,i}$) | 120 | $20 \times 6$ |
| **Total Variables** | **360** | |
| | | |
| **Constraints** | | |
| Demand satisfaction | 20 | $20$ |
| Capacity limits | 120 | $20 \times 6$ |
| Reserve margin | 20 | $20$ |
| Initial capacity | 6 | $6$ |
| Capacity evolution | 114 | $19 \times 6$ |
| Ramp rate (disabled) | 0 | - |
| **Total Constraints** | **~280** | |
| | | |
| **Problem Type** | Linear Program (LP) | All constraints and objectives linear |
| **Solve Time** | <1 minute (HiGHS) | Very fast |
| | <10 seconds (Gurobi) | |

### Computational Complexity

**LP characteristics**:
- No integer variables (continuous relaxation)
- Linear constraints only (no quadratic terms)
- Sparse constraint matrix (most coefficients are zero)
- Well-conditioned (no numerical instability)

**Scalability**: Current model is very tractable and solves quickly even without commercial solvers.

---

## Validation and Results

### Model Validation (Phase 2 & 3)

**Test files**: `test_phase3.py`, `test_prototype.py`

**Validation checks**:
1. ✓ All constraints satisfied (demand, capacity, reserve margin)
2. ✓ Energy balance: Generation ≈ Capacity × CF × 8760
3. ✓ Capacity evolution correct with lead times
4. ✓ Retirements tracked accurately
5. ✓ Objective values match manual calculations

### Benchmark Results (Cost-Optimal, α=1.0)

**Phase 2** (No lead times, no retirements):
- Total cost: $97.26B NPV
- Total emissions: 328.62 MT CO₂
- New capacity: 4.36 GW
- Technology choice: 100% natural gas (cheapest option)
- Solve time: <1 minute

**Phase 3** (With lead times and retirements):
- Total cost: $125.09B NPV (+29% vs Phase 2)
- Total emissions: 507.27 MT CO₂ (+54% vs Phase 2)
- New capacity: 23.75 GW (5.4× more than Phase 2)
- Technology choice: 100% natural gas
- Key insight: **Retirements dominate capacity needs**, not demand growth

**Phase 4 Pareto Frontier** (Multi-objective):
- 10 Pareto-optimal solutions generated
- Cost range: $125.09B to $220.38B
- Emissions range: 507.27 MT to 44.84 MT CO₂
- Trade-off: 91% emissions reduction costs 76% more ($125B → $220B)

---

## Physical Interpretation

### Cost-Optimal Solution (α=1.0)

**Result**: Model chooses 100% natural gas for new capacity

**Why?**
1. **Lowest capital cost**: Gas = $1,500/kW vs Nuclear = $17,500/kW
2. **Reasonable operating cost**: Gas = $55/MWh (higher than others, but offset by low capex)
3. **Fast construction**: 3-year lead time vs 7 years for nuclear
4. **Adequate capacity factor**: 55% (sufficient for meeting demand)

**Problem**: Cost-optimal = emissions disaster (507 MT CO₂)
- Demonstrates need for emissions constraints or carbon pricing
- Pure cost minimization leads to "gas lock-in"

### Emissions-Optimal Solution (α=0.0)

**Result**: Model chooses low-emission technologies (wind, solar, nuclear, hydro)

**Capacity mix**:
- Massive renewable buildout (50-60 GW wind + solar)
- Nuclear expansion (baseload)
- Minimal gas (only for reserve margin)

**Trade-off**:
- Cost: $220.38B (+76% vs cost-optimal)
- Emissions: 44.84 MT CO₂ (-91% vs cost-optimal)
- New capacity: 75 GW (3.2× more than cost-optimal)

**Why more capacity?**
- Low capacity factors (wind 35%, solar 15%) require 3× capacity vs gas (55%)
- High capital costs but very low operating costs
- Near-zero emissions

### Balanced Solution (α=0.5)

**Middle-ground solution**:
- Cost: $139.20B
- Emissions: 55.52 MT CO₂
- Mix: Nuclear + renewables + some gas
- Represents reasonable compromise

---

## Limitations of Annual Model

### 1. No Operational Detail
- **Issue**: Single annual energy value $p_{t,i}$ cannot capture:
  - Hourly demand variations
  - Peak vs off-peak generation
  - Renewable intermittency
  - Plant ramping capability

**Impact**: Model cannot distinguish between:
- Nuclear (inflexible baseload)
- Hydro (flexible load following)

Both evaluated only by annual capacity factor, ignoring operational flexibility.

### 2. Ramp Rate Constraints Disabled
- **Issue**: Annual resolution makes ramp rate constraints meaningless
- **Impact**: Cannot properly value flexible generation
  - Hydro (0.15 MW/min per MW) treated same as nuclear (0.02 MW/min per MW)
  - Gas turbines (peaking) not distinguished from baseload

### 3. Storage Not Modeled
- **Issue**: No intra-day energy storage
- **Impact**: Overestimates feasibility of high renewable penetration
- Hourly model would show need for batteries/pumped hydro

### 4. Transmission Ignored
- **Issue**: Single-node model (all plants and loads in one location)
- **Impact**: Cannot model regional constraints
- Ontario has geographic diversity (hydro in north, demand in south)

### 5. Forced Outages
- **Issue**: Capacity factor is average, doesn't model random outages
- **Impact**: Reserve margin is proxy, but not probabilistic
- More sophisticated: Loss of Load Probability (LOLP)

---

## Comparison: Annual vs Hourly Models

| Aspect | Annual Model (Current) | Hourly Model (Future) |
|--------|------------------------|------------------------|
| **Time resolution** | Annual energy (MWh/year) | Hourly power (MW per hour) |
| **Variables** | 360 | 69,360 (193× more) |
| **Constraints** | ~280 | ~106,700 (381× more) |
| **Ramp constraints** | ❌ Disabled | ✅ Active |
| **Solve time** | <1 min (HiGHS) | 20-30 min (HiGHS) |
| | <10 sec (Gurobi) | 5-10 min (Gurobi) |
| **Operational realism** | Low | High |
| **Flexibility value** | Not captured | Properly valued |
| **Use case** | Strategic planning | Operational planning |
| **Computational cost** | Very low | Moderate |

**When to use annual model**:
- Preliminary analysis
- Rapid iteration on scenarios
- Strategic long-term planning (decades)
- Screening large numbers of technologies

**When to use hourly model**:
- Detailed operational analysis
- Renewable integration studies
- Flexibility requirements
- Publication-quality results
- Policy decisions requiring operational realism

---

## Implementation Files

### Core Model Files
1. **`src/optimization/model.py`** (428 lines)
   - Main `PowerSystemOptimization` class
   - Data loading, model building, solving, results extraction

2. **`src/optimization/variables.py`** (48 lines)
   - Decision variable definitions: $x_{t,i}$, $p_{t,i}$, $N_{t,i}$

3. **`src/optimization/constraints.py`** (151 lines)
   - All constraint functions
   - Demand, capacity, reserve margin, capacity dynamics

4. **`src/optimization/objectives.py`** (105 lines)
   - Cost, emissions, multi-objective functions

### Data Files
1. **`data/processed/plant_parameters.json`**
   - All technology parameters (capex, opex, emissions, etc.)

2. **`data/processed/demand_forecast.csv`**
   - Annual demand and peak demand (2025-2045)

3. **`data/processed/initial_capacity.json`**
   - Starting capacity for each plant type (2025)

### Analysis Files
1. **`src/analysis/pareto.py`** (163 lines)
   - Pareto frontier generation using weighted sum method

2. **`generate_pareto.py`** (142 lines)
   - Script to run multi-objective optimization

### Test Files
1. **`test_phase3.py`** (165 lines)
   - Comprehensive testing of Phase 3 features

2. **`test_prototype.py`**, **`test_full_model.py`**
   - Additional validation tests

---

## Solution Approach

### Single-Objective Optimization

**Algorithm**: Simplex or Interior Point method (LP solver)

**Steps**:
1. Load data (plant parameters, demand forecast, initial capacity)
2. Build Pyomo model (sets, parameters, variables, constraints, objective)
3. Select solver (HiGHS or Gurobi)
4. Solve LP
5. Extract results (capacity schedule, generation mix, costs, emissions)

**Typical solve**:
```
Presolve: 280 constraints → 210 constraints (70 eliminated)
          360 variables → 300 variables (60 eliminated at bounds)
Simplex iterations: 150-165
Solve time: 0.8 seconds (HiGHS)
```

### Multi-Objective Pareto Frontier

**Method**: Weighted sum with varying α

**Algorithm**:
```
1. Solve cost-only (α=1): Get Z_cost_min, Z_emissions_cost
2. Solve emissions-only (α=0): Get Z_emissions_min, Z_cost_emissions
3. Set normalization factors:
   - Z_cost_max = Z_cost_emissions
   - Z_emissions_max = Z_emissions_cost
4. For α in {0.1, 0.2, ..., 0.9}:
   - Build multi-objective with α
   - Solve
   - Record (Z_cost(α), Z_emissions(α))
5. Plot Pareto frontier
```

**Output**: 10-15 Pareto-optimal solutions showing cost-emissions trade-offs

**File**: `src/analysis/pareto.py` implements this algorithm

---

## Key Insights from Annual Model

### 1. Retirements Dominate Capacity Planning
- Demand growth alone: 4.36 GW needed
- With retirements: 23.75 GW needed (5.4× more)
- **Insight**: Replacement of aging capacity is the primary driver, not demand growth

### 2. Cost Optimization Leads to Gas Lock-In
- Pure cost minimization → 100% natural gas expansion
- Result: 507 MT CO₂ emissions (very high)
- **Insight**: Emissions constraints or carbon pricing essential for clean energy transition

### 3. Steep Cost-Emissions Trade-Off
- 91% emissions reduction costs 76% more ($95B additional investment)
- Marginal abatement cost: ~$200-500/ton CO₂
- **Insight**: Deep decarbonization is expensive but feasible

### 4. Long Lead Times Force Early Planning
- Nuclear (7 years): Must decide in 2025 what's needed in 2032
- Hydro (6 years): Plan 6 years ahead
- **Insight**: Short-term decisions have long-term consequences

### 5. Low Capacity Factors Drive Renewable Capacity Needs
- Solar (15% CF): Need 6.7 MW solar to replace 1 MW gas (55% CF)
- Wind (35% CF): Need 1.6 MW wind to replace 1 MW gas
- **Insight**: Renewable-heavy systems require massive capacity buildout

---

## Model Assumptions

### Simplifying Assumptions
1. **Perfect foresight**: Demand known with certainty
2. **Linear relationships**: No economies of scale in construction
3. **Continuous variables**: Can build fractional MW (not integer plants)
4. **Single node**: No transmission constraints
5. **Deterministic**: No weather variability, no forced outages (beyond CF)
6. **Annual resolution**: No sub-annual operational details

### Reasonable Assumptions
1. **Discount rate**: 3.92% real (Canada long-term rate)
2. **Reserve margin**: 15% (industry standard)
3. **Capacity factors**: Based on Ontario historical data
4. **Plant costs**: 2023-2024 NREL and IESO estimates
5. **Retirements**: Simplified but captures ~920 MW/year observed rate

---

## Future Extensions

### Near-Term (Phases 5-6)
1. Sensitivity analysis (discount rate, demand growth, technology costs)
2. Advanced visualizations (capacity timelines, cost breakdowns)
3. Comprehensive testing and documentation

### Long-Term (Phase 7+)
1. **Hourly resolution** (representative days) - most important
2. **Battery storage** as technology option
3. **Transmission constraints** (multi-node model)
4. **Stochastic optimization** (uncertain demand, weather)
5. **Integer variables** (discrete plant sizes)
6. **Carbon pricing** scenarios

---

## References

**Ontario Energy System**:
- IESO Annual Planning Outlook 2023
- Ontario Power Generation reports
- Canada Energy Regulator forecasts

**Capacity Expansion Planning**:
- Integrated Resource Planning (IRP) methodology
- NREL Regional Energy Deployment System (ReEDS)

**Mathematical Programming**:
- Pyomo documentation and user guide
- Linear programming theory (Dantzig, Simplex method)

---

## Summary

The annual resolution model is a **well-designed, computationally efficient LP** that:

✅ **Strengths**:
1. Fast solve times (<1 minute)
2. Captures key drivers: retirements, lead times, reserve margins
3. Multi-objective optimization reveals cost-emissions trade-offs
4. Validated and tested (Phases 1-4 complete)
5. Good for strategic planning and scenario screening

⚠️ **Limitations**:
1. No hourly operational detail (ramp rates disabled)
2. Cannot properly value flexible generation (hydro vs nuclear)
3. Overestimates feasibility of renewables (no intermittency modeling)
4. Single-node (no transmission constraints)

🎯 **Use Cases**:
- Preliminary capacity expansion planning
- Long-term strategic analysis (20+ years)
- Rapid scenario iteration
- Understanding cost-emissions trade-offs at high level

🚀 **Next Step**: Implement hourly resolution model (Phase 7) to add operational realism and activate ramp rate constraints.
