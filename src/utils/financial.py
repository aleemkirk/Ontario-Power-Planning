"""
Financial calculation utilities.

Includes:
- Net Present Value (NPV) calculations
- Discount rate applications
- Annualized cost calculations
"""

import numpy as np
from typing import Union, List


def calculate_npv(cash_flows: Union[List[float], np.ndarray],
                 discount_rate: float) -> float:
    """
    Calculate Net Present Value of cash flows.

    NPV = Σ(CF_t / (1 + r)^t)

    Args:
        cash_flows: List of cash flows by year (year 0 is first element)
        discount_rate: Annual discount rate (e.g., 0.0392 for 3.92%)

    Returns:
        Net present value
    """
    cash_flows = np.array(cash_flows)
    years = np.arange(len(cash_flows))
    discount_factors = 1 / (1 + discount_rate) ** years
    return np.sum(cash_flows * discount_factors)


def calculate_discount_factor(year: int, discount_rate: float) -> float:
    """
    Calculate discount factor for a given year.

    DF = 1 / (1 + r)^t

    Args:
        year: Year (0 = base year)
        discount_rate: Annual discount rate

    Returns:
        Discount factor
    """
    return 1 / (1 + discount_rate) ** year


def annualized_cost(npv: float, n_years: int, discount_rate: float) -> float:
    """
    Convert NPV to equivalent annual cost.

    Uses capital recovery factor:
    AC = NPV × [r(1+r)^n] / [(1+r)^n - 1]

    Args:
        npv: Net present value
        n_years: Number of years
        discount_rate: Annual discount rate

    Returns:
        Annualized cost
    """
    if discount_rate == 0:
        return npv / n_years

    crf = (discount_rate * (1 + discount_rate) ** n_years) / \
          ((1 + discount_rate) ** n_years - 1)
    return npv * crf


def levelized_cost(capital_cost: float, annual_opex: float,
                  annual_generation: float, lifetime: int,
                  discount_rate: float) -> float:
    """
    Calculate Levelized Cost of Electricity (LCOE).

    LCOE = (NPV of costs) / (NPV of generation)

    Args:
        capital_cost: Upfront capital cost ($)
        annual_opex: Annual operating cost ($/year)
        annual_generation: Annual generation (MWh/year)
        lifetime: Plant lifetime (years)
        discount_rate: Annual discount rate

    Returns:
        LCOE ($/MWh)
    """
    # NPV of costs
    opex_cash_flows = [annual_opex] * lifetime
    npv_opex = calculate_npv(opex_cash_flows, discount_rate)
    total_cost_npv = capital_cost + npv_opex

    # NPV of generation
    generation_cash_flows = [annual_generation] * lifetime
    npv_generation = calculate_npv(generation_cash_flows, discount_rate)

    return total_cost_npv / npv_generation if npv_generation > 0 else float('inf')
