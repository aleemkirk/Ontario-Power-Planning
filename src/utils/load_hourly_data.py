"""
Data loading utilities for hourly resolution model.

Functions to load and scale representative days for the full planning horizon,
preparing data for the hourly optimization model.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


def load_representative_days_template(file_path='data/processed/representative_days_12clusters.json'):
    """
    Load 2024 representative days template.

    Args:
        file_path: Path to representative days JSON file

    Returns:
        Dictionary with representative day data
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Representative days file not found: {file_path}\n"
            f"Run 'python data/create_representative_days.py' first to generate it."
        )

    with open(file_path, 'r') as f:
        data = json.load(f)

    print(f"✓ Loaded {data['metadata']['n_clusters']} representative days from {data['metadata']['template_year']}")

    return data


def load_demand_forecast(file_path='data/processed/demand_forecast.csv'):
    """
    Load annual demand forecast for planning horizon.

    Args:
        file_path: Path to demand forecast CSV

    Returns:
        DataFrame with annual_demand and peak_demand by year
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Demand forecast file not found: {file_path}")

    df = pd.read_csv(file_path)

    # Ensure required columns exist
    required_cols = ['year', 'annual_demand', 'peak_demand']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Demand forecast missing columns: {missing_cols}")

    print(f"✓ Loaded demand forecast for {len(df)} years ({df['year'].min()}-{df['year'].max()})")

    return df


def scale_representative_day(rep_day_data: Dict, scale_factor: float) -> Dict:
    """
    Scale a single representative day by a given factor.

    Args:
        rep_day_data: Original representative day data
        scale_factor: Scaling factor to apply

    Returns:
        Scaled representative day data
    """
    scaled_data = {
        'cluster_id': rep_day_data['cluster_id'],
        'representative_date': rep_day_data['representative_date'],
        'weight': rep_day_data['weight'],
        'hourly_demand': [demand * scale_factor for demand in rep_day_data['hourly_demand']],
        'mean_demand': rep_day_data['mean_demand'] * scale_factor,
        'peak_demand': rep_day_data['peak_demand'] * scale_factor,
        'cluster_size': rep_day_data.get('cluster_size', rep_day_data['weight']),
    }

    return scaled_data


def scale_representative_days_for_year(rep_days_template: Dict,
                                       target_year: int,
                                       demand_forecast: pd.DataFrame,
                                       base_year: int = 2024) -> Dict:
    """
    Scale 2024 representative days template to match target year demand.

    Args:
        rep_days_template: Original representative days from base year
        target_year: Year to scale to (e.g., 2025-2045)
        demand_forecast: DataFrame with annual_demand by year
        base_year: Base year for template (default 2024)

    Returns:
        Dictionary of scaled representative days for target year
    """
    # Get base year annual demand (GWh)
    base_demand_row = demand_forecast[demand_forecast['year'] == base_year]
    if len(base_demand_row) == 0:
        # If base year not in forecast, use template year total from validation
        # Calculate from representative days
        base_demand_gwh = sum(
            day['weight'] * sum(day['hourly_demand'])
            for day in rep_days_template.values()
        ) / 1000  # MWh to GWh
        print(f"  Note: Base year {base_year} not in forecast, using calculated value: {base_demand_gwh:.0f} GWh")
    else:
        base_demand_gwh = base_demand_row['annual_demand'].values[0]

    # Get target year annual demand (GWh)
    target_demand_row = demand_forecast[demand_forecast['year'] == target_year]
    if len(target_demand_row) == 0:
        raise ValueError(f"Target year {target_year} not found in demand forecast")

    target_demand_gwh = target_demand_row['annual_demand'].values[0]

    # Calculate scaling factor
    scale_factor = target_demand_gwh / base_demand_gwh

    # Scale each representative day
    scaled_rep_days = {}
    for day_id, day_data in rep_days_template.items():
        scaled_rep_days[int(day_id)] = scale_representative_day(day_data, scale_factor)

    return scaled_rep_days


def create_hourly_demand_dataset(start_year: int = 2025,
                                 end_year: int = 2045,
                                 rep_days_file: str = 'data/processed/representative_days_12clusters.json',
                                 demand_forecast_file: str = 'data/processed/demand_forecast.csv') -> Dict:
    """
    Create complete hourly demand dataset for all years in planning horizon.

    This is the main function to use for loading hourly data into the optimization model.

    Args:
        start_year: First year of planning horizon
        end_year: Last year of planning horizon
        rep_days_file: Path to representative days JSON
        demand_forecast_file: Path to demand forecast CSV

    Returns:
        Dictionary with structure:
        {
            'years': [2025, 2026, ..., 2045],
            'n_rep_days': 12,
            'hours_per_day': 24,
            'rep_days_by_year': {
                2025: {0: {...}, 1: {...}, ..., 11: {...}},
                2026: {0: {...}, 1: {...}, ..., 11: {...}},
                ...
            },
            'metadata': {...}
        }
    """
    print(f"\n[Creating Hourly Demand Dataset: {start_year}-{end_year}]")
    print("-" * 60)

    # Load base data
    rep_days_data = load_representative_days_template(rep_days_file)
    demand_forecast = load_demand_forecast(demand_forecast_file)

    # Filter demand forecast to planning horizon
    demand_forecast = demand_forecast[
        (demand_forecast['year'] >= start_year) &
        (demand_forecast['year'] <= end_year)
    ].copy()

    if len(demand_forecast) != (end_year - start_year + 1):
        print(f"Warning: Demand forecast has {len(demand_forecast)} years, expected {end_year - start_year + 1}")

    # Extract template data
    rep_days_template = rep_days_data['representative_days']
    base_year = rep_days_data['metadata']['template_year']
    n_clusters = rep_days_data['metadata']['n_clusters']

    # Scale for each year
    print(f"\nScaling {n_clusters} representative days for {len(demand_forecast)} years...")

    rep_days_by_year = {}
    for year in range(start_year, end_year + 1):
        rep_days_by_year[year] = scale_representative_days_for_year(
            rep_days_template, year, demand_forecast, base_year
        )

        # Print progress every 5 years
        if (year - start_year) % 5 == 0 or year == end_year:
            annual_total = sum(
                day['weight'] * sum(day['hourly_demand'])
                for day in rep_days_by_year[year].values()
            ) / 1000  # MWh to GWh
            peak = max(
                max(day['hourly_demand'])
                for day in rep_days_by_year[year].values()
            )
            print(f"  {year}: Annual = {annual_total:,.0f} GWh, Peak = {peak:,.0f} MW")

    # Create output structure
    output = {
        'years': list(range(start_year, end_year + 1)),
        'n_rep_days': n_clusters,
        'hours_per_day': 24,
        'rep_days_by_year': rep_days_by_year,
        'metadata': {
            'start_year': start_year,
            'end_year': end_year,
            'n_years': end_year - start_year + 1,
            'template_year': base_year,
            'n_clusters': n_clusters,
            'source_files': {
                'rep_days': str(rep_days_file),
                'demand_forecast': str(demand_forecast_file),
            }
        }
    }

    print(f"\n✓ Hourly demand dataset created successfully")
    print(f"  Total years: {output['metadata']['n_years']}")
    print(f"  Representative days per year: {output['n_rep_days']}")
    print(f"  Hours per day: {output['hours_per_day']}")
    print(f"  Total hourly demand constraints: {output['metadata']['n_years'] * output['n_rep_days'] * output['hours_per_day']:,}")

    return output


def get_hourly_demand(hourly_dataset: Dict, year: int, rep_day_id: int, hour: int) -> float:
    """
    Get demand for specific year, representative day, and hour.

    Args:
        hourly_dataset: Output from create_hourly_demand_dataset()
        year: Year (2025-2045)
        rep_day_id: Representative day ID (0-11 for 12 clusters)
        hour: Hour of day (0-23)

    Returns:
        Demand in MW
    """
    if year not in hourly_dataset['rep_days_by_year']:
        raise ValueError(f"Year {year} not in dataset (available: {min(hourly_dataset['years'])}-{max(hourly_dataset['years'])})")

    if rep_day_id not in hourly_dataset['rep_days_by_year'][year]:
        raise ValueError(f"Rep day {rep_day_id} not in dataset (available: 0-{hourly_dataset['n_rep_days']-1})")

    if not 0 <= hour < 24:
        raise ValueError(f"Hour must be 0-23, got {hour}")

    return hourly_dataset['rep_days_by_year'][year][rep_day_id]['hourly_demand'][hour]


def get_rep_day_weight(hourly_dataset: Dict, rep_day_id: int) -> int:
    """
    Get weight (number of days represented) for a representative day.

    Weights are constant across years (same clustering used).

    Args:
        hourly_dataset: Output from create_hourly_demand_dataset()
        rep_day_id: Representative day ID (0-11 for 12 clusters)

    Returns:
        Weight (number of days this rep day represents)
    """
    # Use first year to get weight (constant across all years)
    first_year = hourly_dataset['years'][0]
    return hourly_dataset['rep_days_by_year'][first_year][rep_day_id]['weight']


def validate_hourly_dataset(hourly_dataset: Dict, demand_forecast: pd.DataFrame) -> Dict:
    """
    Validate that scaled hourly dataset matches demand forecast.

    Args:
        hourly_dataset: Output from create_hourly_demand_dataset()
        demand_forecast: DataFrame with annual_demand by year

    Returns:
        Dictionary with validation results for each year
    """
    print("\n[Validating Hourly Dataset]")
    print("-" * 60)

    validation_results = {}
    max_error = 0.0

    for year in hourly_dataset['years']:
        # Calculate total energy from representative days
        total_from_rep_days = sum(
            day['weight'] * sum(day['hourly_demand'])
            for day in hourly_dataset['rep_days_by_year'][year].values()
        )  # Result in MWh

        # Get forecast annual demand (GWh)
        forecast_row = demand_forecast[demand_forecast['year'] == year]
        if len(forecast_row) == 0:
            print(f"  {year}: Not in forecast, skipping validation")
            continue

        forecast_annual_gwh = forecast_row['annual_demand'].values[0]
        forecast_annual_mwh = forecast_annual_gwh * 1000

        # Calculate error
        error_pct = abs(total_from_rep_days - forecast_annual_mwh) / forecast_annual_mwh * 100
        max_error = max(max_error, error_pct)

        validation_results[year] = {
            'rep_days_total_mwh': total_from_rep_days,
            'forecast_total_mwh': forecast_annual_mwh,
            'error_pct': error_pct,
        }

        # Print every 5 years
        if (year - hourly_dataset['years'][0]) % 5 == 0 or year == hourly_dataset['years'][-1]:
            status = "✓" if error_pct < 1.0 else "⚠"
            print(f"  {status} {year}: Error = {error_pct:.3f}% (Rep days: {total_from_rep_days:,.0f} MWh, Forecast: {forecast_annual_mwh:,.0f} MWh)")

    if max_error < 1.0:
        print(f"\n✓ Validation PASSED: Max error = {max_error:.3f}% (<1%)")
    else:
        print(f"\n⚠ Validation WARNING: Max error = {max_error:.3f}% (>1%)")

    return validation_results


def save_hourly_dataset(hourly_dataset: Dict, output_file: str = 'data/processed/hourly_demand_2025_2045.json'):
    """
    Save hourly dataset to JSON file for later use.

    Args:
        hourly_dataset: Output from create_hourly_demand_dataset()
        output_file: Path to save JSON file
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert integer keys to strings for JSON serialization
    serializable_dataset = hourly_dataset.copy()
    serializable_dataset['rep_days_by_year'] = {
        str(year): {
            str(day_id): day_data
            for day_id, day_data in rep_days.items()
        }
        for year, rep_days in hourly_dataset['rep_days_by_year'].items()
    }

    with open(output_path, 'w') as f:
        json.dump(serializable_dataset, f, indent=2)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n✓ Saved hourly dataset to {output_path} ({file_size_mb:.1f} MB)")


def load_hourly_dataset(input_file: str = 'data/processed/hourly_demand_2025_2045.json') -> Dict:
    """
    Load previously saved hourly dataset from JSON file.

    Args:
        input_file: Path to saved JSON file

    Returns:
        Hourly dataset dictionary
    """
    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Hourly dataset file not found: {input_path}\n"
            f"Run create_hourly_demand_dataset() first to generate it."
        )

    with open(input_path, 'r') as f:
        data = json.load(f)

    # Convert string keys back to integers
    data['rep_days_by_year'] = {
        int(year): {
            int(day_id): day_data
            for day_id, day_data in rep_days.items()
        }
        for year, rep_days in data['rep_days_by_year'].items()
    }

    print(f"✓ Loaded hourly dataset: {data['metadata']['n_years']} years, "
          f"{data['n_rep_days']} representative days")

    return data


# Convenience function for quick setup
def setup_hourly_data(start_year=2025, end_year=2045,
                     force_regenerate=False,
                     save_to_file=True):
    """
    One-line setup function to get hourly data ready for model.

    Args:
        start_year: First year of planning horizon
        end_year: Last year of planning horizon
        force_regenerate: If True, regenerate even if saved file exists
        save_to_file: If True, save generated dataset to file

    Returns:
        Hourly dataset ready for use in optimization model
    """
    saved_file = f'data/processed/hourly_demand_{start_year}_{end_year}.json'

    # Try to load existing file first
    if not force_regenerate and Path(saved_file).exists():
        print(f"Loading existing hourly dataset from {saved_file}...")
        try:
            return load_hourly_dataset(saved_file)
        except Exception as e:
            print(f"Failed to load existing file: {e}")
            print("Regenerating dataset...")

    # Generate new dataset
    dataset = create_hourly_demand_dataset(start_year, end_year)

    # Validate
    demand_forecast = load_demand_forecast()
    validate_hourly_dataset(dataset, demand_forecast)

    # Save to file
    if save_to_file:
        save_hourly_dataset(dataset, saved_file)

    return dataset


if __name__ == "__main__":
    # Example usage: Create and validate hourly dataset for 2025-2045
    print("=" * 60)
    print("Hourly Data Loading Utility - Test Run")
    print("=" * 60)

    # Create hourly dataset
    hourly_data = create_hourly_demand_dataset(start_year=2025, end_year=2045)

    # Validate
    demand_forecast = load_demand_forecast()
    validation = validate_hourly_dataset(hourly_data, demand_forecast)

    # Save to file
    save_hourly_dataset(hourly_data)

    # Test accessor functions
    print("\n[Testing Accessor Functions]")
    print("-" * 60)
    demand_2025_day0_hour0 = get_hourly_demand(hourly_data, 2025, 0, 0)
    demand_2045_day11_hour19 = get_hourly_demand(hourly_data, 2045, 11, 19)
    weight_day0 = get_rep_day_weight(hourly_data, 0)

    print(f"✓ Demand (2025, day 0, hour 0): {demand_2025_day0_hour0:,.0f} MW")
    print(f"✓ Demand (2045, day 11, hour 19): {demand_2045_day11_hour19:,.0f} MW")
    print(f"✓ Weight for day 0: {weight_day0} days")

    print("\n" + "=" * 60)
    print("✓ Hourly data loading utilities test complete!")
    print("=" * 60)
