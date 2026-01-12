# Mathematical Formulation: Hourly Resolution Model
## Ontario Power Plant Optimization

---

## Model Overview

**Type**: Multi-objective Linear Program (LP)
**Time horizon**: 20 years (2025-2045)
**Temporal resolution**: Representative days (12-24 clusters) with hourly dispatch
**Plant types**: Nuclear, Wind, Solar, Natural Gas, Hydro, Biofuel
**Purpose**: Determine optimal power plant capacity expansion to minimize cost and emissions while meeting demand

---

## Sets and Indices

| Symbol | Description | Size |
|--------|-------------|------|
| $t \in T$ | Years in planning horizon | $\|T\| = 20$ |
| $d \in D$ | Representative day clusters | $\|D\| = 12$ (or 24) |
| $h \in H$ | Hours within each representative day | $\|H\| = 24$ |
| $i \in I$ | Plant types | $\|I\| = 6$ |

**Note**: Full annual hourly model would have $h \in \{0, 1, ..., 8759\}$, but representative day approach reduces this to $d \times h$ combinations.

---

## Decision Variables

### Annual Planning Variables
| Variable | Domain | Units | Description |
|----------|--------|-------|-------------|
| $x_{t,i}$ | $\mathbb{R}_+$ | MW | New capacity of plant type $i$ to build in year $t$ |
| $N_{t,i}$ | $\mathbb{R}_+$ | MW | Total operating capacity of plant type $i$ in year $t$ |

### Hourly Operational Variables
| Variable | Domain | Units | Description |
|----------|--------|-------|-------------|
| $p_{t,d,h,i}$ | $\mathbb{R}_+$ | MW | Hourly power output from plant type $i$ in year $t$, day $d$, hour $h$ |
| $v_{t,d,h,i}$ | $\mathbb{R}_+$ | MW | Ramp rate constraint violation (soft constraint slack) |

**Total variables**:
- Annual: $2 \times |T| \times |I| = 2 \times 20 \times 6 = 240$
- Hourly: $2 \times |T| \times |D| \times |H| \times |I| = 2 \times 20 \times 12 \times 24 \times 6 = 69,120$
- **Total: ~69,360 variables**

---

## Parameters

### Plant Technology Parameters
| Parameter | Units | Description |
|-----------|-------|-------------|
| $CapEx_i$ | \$/kW | Capital cost per kW capacity |
| $OpEx_i$ | \$/MWh | Operating cost per MWh generated |
| $MainEx_i$ | \$/MW/year | Annual maintenance cost per MW capacity |
| $EmissionFactor_i$ | tons CO₂/MWh | Emissions per MWh generated |
| $CF_i$ | - | Capacity factor (fraction of time at full output) |
| $RampRate_i$ | MW/min per MW | Maximum ramp rate (fraction per minute) |
| $LeadTime_i$ | years | Construction lead time |
| $Lifespan_i$ | years | Plant operational lifetime |

### System Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| $r$ | 3.92% | Real discount rate |
| $RM$ | 15% | Reserve margin above peak demand |
| $w_d$ | days | Weight of representative day $d$ (number of days it represents) |
| $RampPenalty$ | \$/MW | Penalty for ramp constraint violations (e.g., \$1000/MW) |

### Time-Varying Parameters
| Parameter | Units | Description |
|-----------|-------|-------------|
| $Demand_{t,d,h}$ | MW | Electricity demand in year $t$, rep day $d$, hour $h$ |
| $PeakDemand_t$ | MW | Annual peak demand in year $t$ |
| $InitialCapacity_i$ | MW | Existing capacity of plant type $i$ in year $t_0$ |
| $Retirement_{t,i}$ | MW | Capacity retiring in year $t$ for plant type $i$ |
| $\delta_\tau$ | - | Discount factor for year index $\tau$: $\delta_\tau = \frac{1}{(1+r)^\tau}$ |

**Note**: $\sum_{d \in D} w_d = 365$ (representative days cover all days of the year)

---

## Objective Functions

### 1. Total System Cost (Minimize)

$$
\min Z_{cost} = \sum_{\tau=0}^{|T|-1} \delta_\tau \left[ \underbrace{\sum_{i \in I} x_{t_0+\tau,i} \cdot CapEx_i \cdot 1000}_{\text{Capital Costs}} + \underbrace{\sum_{d \in D} \sum_{h \in H} \sum_{i \in I} p_{t_0+\tau,d,h,i} \cdot w_d \cdot OpEx_i}_{\text{Operating Costs}} + \underbrace{\sum_{i \in I} N_{t_0+\tau,i} \cdot MainEx_i}_{\text{Maintenance Costs}} + \underbrace{\sum_{d \in D} \sum_{h \in H} \sum_{i \in I} v_{t_0+\tau,d,h,i} \cdot w_d \cdot RampPenalty}_{\text{Ramp Violation Penalty}} \right]
$$

**Components**:
- **Capital costs**: Discounted NPV of new capacity investments (×1000 converts $/kW to $/MW)
- **Operating costs**: Discounted cost of generation, weighted by representative day frequency
- **Maintenance costs**: Discounted annual maintenance of operating capacity
- **Ramp penalty**: Penalty for violating ramp rate constraints (soft constraint approach)

**Discount factor**: $\delta_\tau = \frac{1}{(1+r)^\tau}$ where $\tau = t - t_0$ (years from start)

### 2. Total Carbon Emissions (Minimize)

$$
\min Z_{emissions} = \sum_{t \in T} \sum_{d \in D} \sum_{h \in H} \sum_{i \in I} p_{t,d,h,i} \cdot w_d \cdot EmissionFactor_i
$$

**Note**: Representative day weights $w_d$ ensure emissions are properly scaled to annual totals.

### 3. Multi-Objective Weighted Sum

$$
\min Z_{multi} = \alpha \cdot \frac{Z_{cost}}{Z_{cost}^{max}} + (1-\alpha) \cdot \frac{Z_{emissions}}{Z_{emissions}^{max}}
$$

where:
- $\alpha \in [0, 1]$: Weight parameter
  - $\alpha = 1$: Pure cost minimization
  - $\alpha = 0$: Pure emissions minimization
  - $0 < \alpha < 1$: Trade-off between objectives
- $Z_{cost}^{max}$: Normalization factor (cost from emissions-optimal solution)
- $Z_{emissions}^{max}$: Normalization factor (emissions from cost-optimal solution)

**Pareto frontier generation**: Solve for multiple $\alpha$ values (e.g., $\alpha \in \{0.0, 0.1, 0.2, ..., 1.0\}$) to generate trade-off curve.

---

## Constraints

### 1. Hourly Demand Satisfaction

$$
\sum_{i \in I} p_{t,d,h,i} \geq Demand_{t,d,h} \quad \forall t \in T, \forall d \in D, \forall h \in H
$$

**Description**: Total generation must meet or exceed demand every hour of every representative day.

**Number of constraints**: $|T| \times |D| \times |H| = 20 \times 12 \times 24 = 5,760$

---

### 2. Hourly Capacity Constraint

$$
p_{t,d,h,i} \leq N_{t,i} \cdot CF_i \quad \forall t \in T, \forall d \in D, \forall h \in H, \forall i \in I
$$

**Description**: Hourly generation cannot exceed available capacity adjusted by capacity factor.

**Interpretation**:
- $N_{t,i}$ is the installed capacity (nameplate, MW)
- $CF_i$ is the maximum fraction of time the plant can operate
- For nuclear ($CF = 0.90$): Can generate up to 90% of capacity each hour
- For solar ($CF = 0.15$): Can generate up to 15% of capacity (averaged over all hours)

**Number of constraints**: $|T| \times |D| \times |H| \times |I| = 20 \times 12 \times 24 \times 6 = 34,560$

---

### 3. Ramp Rate Constraints (NEW - ACTIVATED IN HOURLY MODEL)

#### 3a. Ramp Up Constraint (Soft)

$$
p_{t,d,h,i} - p_{t,d,h-1,i} \leq RampRate_i \cdot N_{t,i} \cdot 60 + v_{t,d,h,i} \quad \forall t \in T, \forall d \in D, \forall h \in H \setminus \{0\}, \forall i \in I
$$

#### 3b. Ramp Down Constraint (Soft)

$$
p_{t,d,h-1,i} - p_{t,d,h,i} \leq RampRate_i \cdot N_{t,i} \cdot 60 + v_{t,d,h,i} \quad \forall t \in T, \forall d \in D, \forall h \in H \setminus \{0\}, \forall i \in I
$$

**Description**: Limits the rate of change in power output between consecutive hours.

**Interpretation**:
- $RampRate_i$ is in MW/min per MW of capacity
- Multiplied by 60 to convert to MW/hour
- $v_{t,d,h,i}$ is slack variable allowing violations (penalized in objective)
- Hard constraint version: Remove $v_{t,d,h,i}$ term (may cause infeasibility)

**Example**:
- Nuclear with 1000 MW capacity: $RampRate = 0.02 \times 1000 \times 60 = 1200$ MW/hour max ramp
- Hydro with 1000 MW capacity: $RampRate = 0.15 \times 1000 \times 60 = 9000$ MW/hour max ramp (but limited by capacity)

**Number of constraints**: $2 \times |T| \times |D| \times (|H|-1) \times |I| = 2 \times 20 \times 12 \times 23 \times 6 = 66,240$

**Note**: $h=0$ (first hour) skipped because no previous hour to compare against.

---

### 4. Annual Reserve Margin

$$
\sum_{i \in I} N_{t,i} \geq (1 + RM) \cdot PeakDemand_t \quad \forall t \in T
$$

**Description**: Total installed capacity must exceed peak demand by reserve margin (15%).

**Purpose**: Ensures sufficient capacity for reliability, forced outages, and maintenance.

**Number of constraints**: $|T| = 20$

---

### 5. Capacity Dynamics with Lead Times and Retirements

#### 5a. Initial Capacity

$$
N_{t_0,i} = InitialCapacity_i \quad \forall i \in I
$$

#### 5b. Capacity Evolution

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
- Previous year's capacity
- New capacity coming online (built $LeadTime_i$ years ago)
- Retirements based on plant age and lifespan

**Example**:
- Nuclear plant ($LeadTime = 7$ years) built in 2025 becomes operational in 2032
- Forces planning ahead: Must decide in 2025 what's needed in 2032

**Number of constraints**: $|I| + (|T|-1) \times |I| = 6 + 19 \times 6 = 120$

---

### 6. Non-Negativity

$$
\begin{align}
x_{t,i} &\geq 0 \quad \forall t \in T, \forall i \in I \\
N_{t,i} &\geq 0 \quad \forall t \in T, \forall i \in I \\
p_{t,d,h,i} &\geq 0 \quad \forall t \in T, \forall d \in D, \forall h \in H, \forall i \in I \\
v_{t,d,h,i} &\geq 0 \quad \forall t \in T, \forall d \in D, \forall h \in H, \forall i \in I
\end{align}
$$

**Description**: All decision variables must be non-negative (capacity and generation cannot be negative).

**Implementation**: Enforced by variable domain specification ($\mathbb{R}_+$).

---

## Problem Size Summary

### Hourly Model (12 Representative Days)

| Component | Count | Calculation |
|-----------|-------|-------------|
| **Variables** | | |
| Annual planning ($x_{t,i}$, $N_{t,i}$) | 240 | $2 \times 20 \times 6$ |
| Hourly dispatch ($p_{t,d,h,i}$) | 34,560 | $20 \times 12 \times 24 \times 6$ |
| Ramp violations ($v_{t,d,h,i}$) | 34,560 | $20 \times 12 \times 24 \times 6$ |
| **Total Variables** | **69,360** | |
| | | |
| **Constraints** | | |
| Demand satisfaction | 5,760 | $20 \times 12 \times 24$ |
| Capacity limits | 34,560 | $20 \times 12 \times 24 \times 6$ |
| Ramp up | 33,120 | $20 \times 12 \times 23 \times 6$ |
| Ramp down | 33,120 | $20 \times 12 \times 23 \times 6$ |
| Reserve margin | 20 | $20$ |
| Initial capacity | 6 | $6$ |
| Capacity evolution | 114 | $19 \times 6$ |
| **Total Constraints** | **106,700** | |

### Comparison to Annual Model

| Metric | Annual Model | Hourly Model (12 rep days) | Increase Factor |
|--------|--------------|----------------------------|-----------------|
| Variables | 360 | 69,360 | **193×** |
| Constraints | ~500 | ~106,700 | **213×** |
| Solve time (HiGHS) | <1 min | 20-30 min | **20-30×** |
| Solve time (Gurobi) | <10 sec | 5-10 min | **30-60×** |

---

## Key Differences from Annual Model

### 1. Time Resolution
- **Annual**: Single annual energy value $p_{t,i}$ (MWh/year)
- **Hourly**: Hourly power output $p_{t,d,h,i}$ (MW per hour)

### 2. Demand Constraint
- **Annual**: $\sum_i p_{t,i} \geq AnnualDemand_t$ (annual energy balance)
- **Hourly**: $\sum_i p_{t,d,h,i} \geq Demand_{t,d,h}$ (hourly power balance)

### 3. Capacity Constraint
- **Annual**: $p_{t,i} \leq N_{t,i} \times CF_i \times 8760$ (energy over year)
- **Hourly**: $p_{t,d,h,i} \leq N_{t,i} \times CF_i$ (power each hour)

### 4. Ramp Rate Constraints
- **Annual**: DISABLED (not applicable at annual resolution)
- **Hourly**: ACTIVE (constrains hour-to-hour changes)

### 5. Objective Functions
- **Annual**: Sum over years and plant types
- **Hourly**: Sum over years, representative days, hours, and plant types (with day weights)

---

## Representative Day Clustering

### Motivation
Full hourly model (8,760 hours/year) would have:
- Variables: $2 \times 20 \times 8760 \times 6 = 2,102,400$
- Constraints: $\sim 3,500,000$
- Solve time: HOURS or INFEASIBLE

**Solution**: Cluster 365 days → 12-24 representative days

### Clustering Algorithm

**Input**: 8,760 hourly demand values for one year

**Steps**:
1. Reshape to 365 days × 24 hours
2. Extract features for each day:
   - Mean demand
   - Peak demand
   - Time of peak
   - Standard deviation
   - Season (winter/spring/summer/fall)
3. Apply K-means clustering ($k = 12$ or $k = 24$)
4. For each cluster, select most representative day (closest to cluster center)
5. Weight each representative day by cluster size (number of days it represents)

**Output**:
- $D = 12$ representative days
- Weights: $w_d$ (typically $w_d \approx 30$ days each)
- Constraint: $\sum_{d=1}^{12} w_d = 365$

**Validation**:
- Energy balance: $\sum_d w_d \cdot \sum_h Demand_{d,h} \approx \sum_{d=1}^{365} \sum_h Demand_{d,h}$
- Peak preserved: $\max_{d,h} Demand_{d,h} \approx \max_{d=1...365,h} Demand_{d,h}$
- Typical accuracy: R² > 0.95

### Example Clustering (12 clusters)

| Cluster | Description | Weight (days) | Avg Demand (MW) |
|---------|-------------|---------------|-----------------|
| 1 | Winter weekday peak | 60 | 22,000 |
| 2 | Winter weekend | 30 | 19,000 |
| 3 | Spring weekday | 45 | 18,500 |
| 4 | Spring weekend | 20 | 16,000 |
| 5 | Summer weekday peak | 55 | 23,000 |
| 6 | Summer weekend | 25 | 18,500 |
| 7 | Fall weekday | 50 | 19,500 |
| 8 | Fall weekend | 25 | 17,000 |
| 9 | Extreme peak day | 10 | 24,500 |
| 10 | Extreme low day | 10 | 15,000 |
| 11 | High renewable gen day | 20 | 17,500 |
| 12 | Low renewable gen day | 15 | 20,500 |
| **Total** | | **365** | |

---

## Physical Interpretation

### Ramp Rate Constraints

**Operational significance**: Different technologies have different flexibility:

| Plant Type | Ramp Rate (MW/min per MW) | 1 GW Plant Max Ramp (MW/hour) | Operational Role |
|------------|---------------------------|--------------------------------|------------------|
| Hydro | 0.15 | 9,000* | Load following, balancing |
| Solar | 0.10 | 6,000* | Fast response (weather-dependent) |
| Wind | 0.05 | 3,000 | Moderate flexibility |
| Gas | 0.04 | 2,400 | Peaking, load following |
| Nuclear | 0.02 | 1,200 | Baseload (inflexible) |
| Biofuel | 0.01 | 600 | Baseload (very slow) |

*Actual ramp limited by capacity (max 1 GW output)

**Implication**:
- Nuclear operates as baseload (constant output)
- Hydro handles morning/evening ramps
- Gas provides intermediate flexibility
- Model will value flexible capacity more than annual model

### Capacity Factor vs Ramping

**Capacity factor**: Average utilization over time
- Nuclear: 90% (runs almost constantly)
- Solar: 15% (only during daytime, weather-dependent)

**Ramping capability**: How quickly output can change
- Independent of capacity factor
- Critical for integrating renewables
- Determines operational flexibility value

**Example**:
- Nuclear: High capacity factor (90%), low ramping (0.02)
- Hydro: Medium capacity factor (50%), high ramping (0.15)
- In hourly model, hydro becomes more valuable despite lower capacity factor

---

## Solution Approach

### Pareto Frontier Generation

**Weighted sum method**:

1. Solve cost-only optimization ($\alpha = 1$)
   - Get $Z_{cost}^{min}$ and $Z_{emissions}^{cost\_opt}$
   - Set $Z_{emissions}^{max} = Z_{emissions}^{cost\_opt}$

2. Solve emissions-only optimization ($\alpha = 0$)
   - Get $Z_{emissions}^{min}$ and $Z_{cost}^{emissions\_opt}$
   - Set $Z_{cost}^{max} = Z_{cost}^{emissions\_opt}$

3. For $\alpha \in \{0.1, 0.2, ..., 0.9\}$:
   - Solve multi-objective with normalized objectives
   - Record $(Z_{cost}(\alpha), Z_{emissions}(\alpha))$

4. Plot Pareto frontier: Cost vs Emissions

**Alternative: ε-constraint method**:
- Fix emissions constraint: $Z_{emissions} \leq \epsilon$
- Minimize cost: $\min Z_{cost}$
- Vary $\epsilon$ from $Z_{emissions}^{min}$ to $Z_{emissions}^{max}$
- More computationally expensive but finds true Pareto optimal solutions

---

## Expected Results

### Qualitative Changes (Annual → Hourly)

**Capacity Mix**:
- ↓ Nuclear (10-20%): Inflexible baseload less valuable
- ↑ Hydro (30-50%): Fast ramping highly valued
- ↑ Gas (20-30%): Flexible peaking important
- ≈ Wind/Solar: Weather-driven, not selected for flexibility

**Operational Patterns**:
- Nuclear: 85-90% capacity factor (constant baseload)
- Hydro: 20-80% capacity factor (high variability for load following)
- Gas: Peaks during ramps (morning 6-9am, evening 5-8pm)
- Solar: Follows sunlight (peaks 11am-2pm)

**Costs**:
- Annual model: $125B (cost-optimal)
- Hourly model: $138-145B (10-15% higher)
- Reason: Ramp constraints require more expensive flexible capacity

**Emissions**:
- Similar overall emissions for comparable capacity mixes
- Hourly model may show slight increase if more gas is built for flexibility

---

## Validation Checks

### Energy Balance
$$
\sum_{d \in D} w_d \cdot \sum_{h \in H} \sum_{i \in I} p_{t,d,h,i} \approx \sum_{i \in I} N_{t,i} \cdot CF_i \cdot 8760
$$
Hourly generation (scaled by day weights) should equal annual capacity-based generation.

### Ramp Rate Feasibility
$$
|p_{t,d,h,i} - p_{t,d,h-1,i}| \leq RampRate_i \cdot N_{t,i} \cdot 60 + v_{t,d,h,i}
$$
Check $v_{t,d,h,i} < 0.05 \times RampRate_i \cdot N_{t,i} \cdot 60$ (violations < 5% acceptable for soft constraints).

### Reserve Margin
$$
\frac{\sum_{i} N_{t,i}}{PeakDemand_t} \geq 1.15
$$
Verify reserve margin is maintained.

### Physical Realism
- Nuclear capacity factor: 80-95%
- Solar capacity factor: 10-20%
- Hydro shows high variability (not constant)
- Solar generation peaks during mid-day hours

---

## Implementation Notes

### Solver Selection
- **HiGHS**: Open-source, good for development, 20-30 min solve time
- **Gurobi**: Commercial (free academic license), 5-10 min solve time (recommended for production)

### Warm Starting
Initialize hourly model with annual model solution:
- $N_{t,i}$, $x_{t,i}$: Copy from annual model
- $p_{t,d,h,i}$: Distribute annual generation uniformly across hours (or use demand profile)

### Soft vs Hard Ramp Constraints
- **Hard constraints**: May cause infeasibility if demand ramps exceed available ramp capacity
- **Soft constraints** (recommended): Allow violations with high penalty ($1000/MW), ensures feasibility
- **Validation**: Check that violations are minimal (<5% of cases)

### Representative Day Selection
- **12 days**: Fast solve (10 min), 2-3% error, recommended for iteration
- **24 days**: Slower solve (25 min), 1-2% error, recommended for publication results

---

## References

**Capacity Expansion Planning**:
- Palmintier & Webster (2011). "Impact of unit commitment constraints on generation expansion planning with renewables." *IEEE Transactions on Power Systems*.

**Representative Days**:
- Nahmmacher et al. (2016). "Carpe diem: A novel approach to select representative days for long-term power system modeling." *Energy*.

**Multi-Objective Optimization**:
- Mavrotas (2009). "Effective implementation of the ε-constraint method in multi-objective mathematical programming problems." *Applied Mathematics and Computation*.

---

## Summary

The hourly resolution model extends the annual model by:
1. ✓ Adding hourly dispatch variables $p_{t,d,h,i}$
2. ✓ Activating ramp rate constraints (previously disabled)
3. ✓ Using representative day clustering for computational tractability
4. ✓ Properly valuing operational flexibility (hydro, gas turbines)
5. ✓ Enabling realistic operational analysis (peak hours, ramping events)

**Trade-offs**:
- **Complexity**: 193× more variables, 213× more constraints
- **Solve time**: 20-30× longer (but still tractable)
- **Realism**: Much more accurate representation of grid operations
- **Insights**: Shows when and how fast plants ramp, not just annual totals

**Result**: A more realistic and policy-relevant power system planning model.
