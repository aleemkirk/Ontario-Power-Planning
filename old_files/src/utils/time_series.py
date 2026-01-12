"""
Time series utilities for demand and generation profiles.

Handles:
- Demand profile generation
- Temporal aggregation
- Peak demand calculation
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


def generate_annual_demand_profile(base_demand: float,
                                   growth_rate: float,
                                   years: range) -> pd.Series:
    """
    Generate annual demand forecast with growth rate.

    Demand[t] = BaseLoad × (1 + g)^t

    Args:
        base_demand: Base year demand (GWh)
        growth_rate: Annual growth rate (e.g., 0.022 for 2.2%)
        years: Range of years

    Returns:
        Series indexed by year with demand values
    """
    year_list = list(years)
    n_years = len(year_list)
    demand = base_demand * (1 + growth_rate) ** np.arange(n_years)
    return pd.Series(demand, index=year_list)


def generate_hourly_profile(annual_demand: float,
                            profile_type: str = 'typical') -> np.ndarray:
    """
    Generate hourly demand profile for a year (8760 hours).

    Args:
        annual_demand: Total annual demand (GWh)
        profile_type: 'typical', 'summer_peak', or 'winter_peak'

    Returns:
        Array of 8760 hourly demand values (MW)
    """
    # Convert GWh to MWh
    annual_mwh = annual_demand * 1000

    # Simple sinusoidal profile for prototype
    # TODO: Replace with actual Ontario demand profile
    hours = np.arange(8760)
    avg_demand = annual_mwh / 8760

    if profile_type == 'typical':
        # Daily variation (24h) + seasonal variation (8760h)
        daily_factor = 0.15 * np.sin(2 * np.pi * hours / 24)
        seasonal_factor = 0.2 * np.sin(2 * np.pi * hours / 8760)
        hourly_demand = avg_demand * (1 + daily_factor + seasonal_factor)

    else:
        hourly_demand = np.full(8760, avg_demand)

    return hourly_demand


def calculate_peak_demand(hourly_profile: np.ndarray) -> float:
    """
    Calculate peak demand from hourly profile.

    Args:
        hourly_profile: Array of hourly demand values (MW)

    Returns:
        Peak demand (MW)
    """
    return np.max(hourly_profile)


def aggregate_to_monthly(hourly_profile: np.ndarray) -> np.ndarray:
    """
    Aggregate hourly profile to monthly averages.

    Args:
        hourly_profile: Array of 8760 hourly values

    Returns:
        Array of 12 monthly average values
    """
    # Approximate hours per month (simplified)
    hours_per_month = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]

    monthly = []
    hour_idx = 0
    for hours in hours_per_month:
        month_avg = np.mean(hourly_profile[hour_idx:hour_idx + hours])
        monthly.append(month_avg)
        hour_idx += hours

    return np.array(monthly)


def create_representative_days(hourly_profile: np.ndarray,
                              n_days: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create representative days from hourly profile.

    Useful for reducing model size while maintaining temporal characteristics.

    Args:
        hourly_profile: Array of 8760 hourly values
        n_days: Number of representative days to create

    Returns:
        Tuple of (representative_profiles, weights)
        - representative_profiles: (n_days, 24) array
        - weights: Array of weights summing to 365
    """
    # TODO: Implement k-means clustering for representative days
    # For now, return simple monthly representatives
    n_days = min(n_days, 12)
    profiles = []
    weights = []

    days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    for month in range(n_days):
        start_hour = sum([d * 24 for d in days_per_month[:month]])
        month_hours = days_per_month[month] * 24
        month_data = hourly_profile[start_hour:start_hour + month_hours]

        # Average day for this month
        avg_day = np.mean(month_data.reshape(-1, 24), axis=0)
        profiles.append(avg_day)
        weights.append(days_per_month[month])

    return np.array(profiles), np.array(weights)
