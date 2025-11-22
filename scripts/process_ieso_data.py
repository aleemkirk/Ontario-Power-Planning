"""
Process and clean IESO hourly demand data.

Reads raw IESO CSV files, cleans headers, parses dates, and creates
a properly formatted hourly demand dataset for model use.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime


def parse_ieso_csv(file_path):
    """
    Parse IESO CSV file which has metadata headers.

    IESO format:
    - Row 0: \\Hourly Demand Report,,,
    - Row 1: \\Created at [timestamp],,,
    - Row 2: \\For [year],,,
    - Row 3: Column headers (Date, Hour, Market Demand, Ontario Demand)
    - Row 4+: Actual data

    Args:
        file_path: Path to raw IESO CSV file

    Returns:
        DataFrame with cleaned data
    """
    # Read CSV skipping metadata rows (skip first 3 rows)
    df = pd.read_csv(file_path, skiprows=3)

    # Fix column names (remove leading backslashes if any)
    df.columns = df.columns.str.strip().str.replace('\\\\', '')

    # Remove any remaining metadata rows
    df = df[df['Date'].notna()].copy()
    df = df[df['Date'] != 'Date'].copy()  # Remove duplicate headers

    return df


def process_all_ieso_files(input_dir='data/raw/ieso', start_year=2002, end_year=2025):
    """
    Process all IESO yearly files into clean dataset.

    Args:
        input_dir: Directory containing raw IESO CSV files
        start_year: First year to process
        end_year: Last year to process

    Returns:
        DataFrame with cleaned hourly demand data
    """
    input_path = Path(input_dir)
    all_data = []

    print(f"Processing IESO data from {start_year} to {end_year}...")
    print("-" * 60)

    for year in range(start_year, end_year + 1):
        file_path = input_path / f"PUB_Demand_{year}.csv"

        if not file_path.exists():
            print(f"✗ {year}: File not found, skipping")
            continue

        try:
            df = parse_ieso_csv(file_path)

            # Validate data
            if len(df) == 0:
                print(f"✗ {year}: No data, skipping")
                continue

            print(f"✓ {year}: {len(df):,} rows ({df['Date'].min()} to {df['Date'].max()})")
            all_data.append(df)

        except Exception as e:
            print(f"✗ {year}: Error - {e}")
            continue

    # Combine all years
    if not all_data:
        raise ValueError("No data files successfully processed")

    combined_df = pd.concat(all_data, ignore_index=True)

    # Clean and standardize
    print("\n[Cleaning and standardizing data]")

    # Convert data types
    combined_df['Date'] = pd.to_datetime(combined_df['Date'])
    combined_df['Hour'] = combined_df['Hour'].astype(int)
    combined_df['Market Demand'] = pd.to_numeric(combined_df['Market Demand'], errors='coerce')
    combined_df['Ontario Demand'] = pd.to_numeric(combined_df['Ontario Demand'], errors='coerce')

    # Sort by date and hour
    combined_df = combined_df.sort_values(['Date', 'Hour']).reset_index(drop=True)

    # Remove duplicates
    original_rows = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['Date', 'Hour'], keep='last')
    duplicates_removed = original_rows - len(combined_df)
    print(f"✓ Removed {duplicates_removed} duplicate rows")

    # Create datetime column (combine date and hour)
    combined_df['DateTime'] = combined_df.apply(
        lambda row: pd.Timestamp(row['Date'].date()) + pd.Timedelta(hours=row['Hour'] - 1),
        axis=1
    )

    # Reorder columns
    combined_df = combined_df[['DateTime', 'Date', 'Hour', 'Ontario Demand', 'Market Demand']]

    print(f"✓ Final dataset: {len(combined_df):,} rows")
    print(f"✓ Date range: {combined_df['DateTime'].min()} to {combined_df['DateTime'].max()}")

    return combined_df


def validate_data(df):
    """
    Validate cleaned IESO data for quality issues.

    Args:
        df: Cleaned DataFrame

    Returns:
        Dictionary of validation results
    """
    print("\n[Data Validation]")

    results = {}

    # Check for missing values
    missing_demand = df['Ontario Demand'].isna().sum()
    missing_pct = (missing_demand / len(df)) * 100
    print(f"Missing demand values: {missing_demand} ({missing_pct:.2f}%)")
    results['missing_values'] = missing_demand

    # Check for negative values
    negative = (df['Ontario Demand'] < 0).sum()
    print(f"Negative demand values: {negative}")
    results['negative_values'] = negative

    # Check for outliers (demand > 30,000 MW or < 10,000 MW)
    outliers_high = (df['Ontario Demand'] > 30000).sum()
    outliers_low = (df['Ontario Demand'] < 10000).sum()
    print(f"Outliers (>30,000 MW): {outliers_high}")
    print(f"Outliers (<10,000 MW): {outliers_low}")
    results['outliers_high'] = outliers_high
    results['outliers_low'] = outliers_low

    # Check for gaps in time series
    df_sorted = df.sort_values('DateTime').copy()
    df_sorted['time_diff'] = df_sorted['DateTime'].diff()
    expected_diff = pd.Timedelta(hours=1)
    gaps = (df_sorted['time_diff'] > expected_diff).sum()
    print(f"Gaps in hourly sequence: {gaps}")
    results['gaps'] = gaps

    # Basic statistics
    print(f"\n[Demand Statistics]")
    print(f"Mean: {df['Ontario Demand'].mean():.0f} MW")
    print(f"Median: {df['Ontario Demand'].median():.0f} MW")
    print(f"Min: {df['Ontario Demand'].min():.0f} MW")
    print(f"Max: {df['Ontario Demand'].max():.0f} MW")
    print(f"Std Dev: {df['Ontario Demand'].std():.0f} MW")

    results['statistics'] = {
        'mean': df['Ontario Demand'].mean(),
        'median': df['Ontario Demand'].median(),
        'min': df['Ontario Demand'].min(),
        'max': df['Ontario Demand'].max(),
        'std': df['Ontario Demand'].std()
    }

    return results


def plot_demand_overview(df, output_path='results/figures/ieso_demand_overview.png'):
    """
    Create overview plots of historical demand data.

    Args:
        df: Cleaned DataFrame
        output_path: Path to save figure
    """
    print(f"\n[Creating visualizations]")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Plot 1: Annual average demand
    df_copy = df.copy()
    df_copy['Year'] = df_copy['DateTime'].dt.year
    annual_avg = df_copy.groupby('Year')['Ontario Demand'].mean()

    axes[0].plot(annual_avg.index, annual_avg.values, marker='o', linewidth=2)
    axes[0].set_title('Annual Average Ontario Electricity Demand (2002-2025)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Year')
    axes[0].set_ylabel('Average Demand (MW)')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Annual peak demand
    annual_peak = df_copy.groupby('Year')['Ontario Demand'].max()

    axes[1].plot(annual_peak.index, annual_peak.values, marker='o', linewidth=2, color='red')
    axes[1].set_title('Annual Peak Ontario Electricity Demand (2002-2025)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Year')
    axes[1].set_ylabel('Peak Demand (MW)')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Recent year hourly profile (2024)
    df_2024 = df_copy[df_copy['Year'] == 2024].copy()
    if len(df_2024) > 0:
        axes[2].plot(df_2024['DateTime'], df_2024['Ontario Demand'], linewidth=0.5, alpha=0.7)
        axes[2].set_title('Hourly Ontario Demand Profile - 2024', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Date')
        axes[2].set_ylabel('Demand (MW)')
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved figure to {output_file}")

    plt.close()


def save_processed_data(df, output_file='data/processed/ieso_hourly_demand_2002_2025.csv'):
    """
    Save cleaned data to processed directory.

    Args:
        df: Cleaned DataFrame
        output_file: Output file path
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved processed data to {output_path}")

    # Also save summary statistics
    summary_file = output_path.parent / 'ieso_demand_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("IESO Hourly Demand Data Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Date range: {df['DateTime'].min()} to {df['DateTime'].max()}\n")
        f.write(f"Total hours: {len(df):,}\n")
        f.write(f"Total years: {df['DateTime'].dt.year.nunique()}\n\n")
        f.write("Demand Statistics (MW):\n")
        f.write(f"  Mean:   {df['Ontario Demand'].mean():,.0f}\n")
        f.write(f"  Median: {df['Ontario Demand'].median():,.0f}\n")
        f.write(f"  Min:    {df['Ontario Demand'].min():,.0f}\n")
        f.write(f"  Max:    {df['Ontario Demand'].max():,.0f}\n")
        f.write(f"  Std:    {df['Ontario Demand'].std():,.0f}\n\n")
        f.write("Data Quality:\n")
        f.write(f"  Missing values: {df['Ontario Demand'].isna().sum()}\n")
        f.write(f"  Negative values: {(df['Ontario Demand'] < 0).sum()}\n")

    print(f"✓ Saved summary to {summary_file}")


if __name__ == "__main__":
    # Process all IESO files
    df = process_all_ieso_files(start_year=2002, end_year=2025)

    # Validate data
    validation_results = validate_data(df)

    # Create visualizations
    plot_demand_overview(df)

    # Save processed data
    save_processed_data(df)

    print("\n" + "=" * 60)
    print("✓ IESO data processing complete!")
    print("=" * 60)
