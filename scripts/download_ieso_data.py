"""
Download historical hourly demand data from IESO.

Downloads Ontario electricity demand data from 2002-2025 from the IESO public reports.
Data source: https://reports-public.ieso.ca/public/Demand/

Output: CSV files saved to data/raw/ieso/
"""

import requests
import time
from pathlib import Path
import pandas as pd


def download_ieso_demand_data(start_year=2002, end_year=2025, output_dir='data/raw/ieso'):
    """
    Download IESO hourly demand data for specified year range.

    Args:
        start_year: First year to download (default 2002)
        end_year: Last year to download (default 2025)
        output_dir: Directory to save downloaded files
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    base_url = "https://reports-public.ieso.ca/public/Demand/"

    downloaded_files = []
    failed_downloads = []

    print(f"Downloading IESO demand data from {start_year} to {end_year}...")
    print(f"Output directory: {output_path.absolute()}")
    print("-" * 60)

    for year in range(start_year, end_year + 1):
        filename = f"PUB_Demand_{year}.csv"
        url = base_url + filename
        output_file = output_path / filename

        # Skip if already downloaded
        if output_file.exists():
            print(f"✓ {year}: Already exists, skipping")
            downloaded_files.append(output_file)
            continue

        try:
            print(f"⬇ {year}: Downloading...", end=" ", flush=True)
            response = requests.get(url, timeout=30)
            response.raise_for_status()  # Raise exception for bad status codes

            # Save to file
            with open(output_file, 'wb') as f:
                f.write(response.content)

            # Validate it's a valid CSV
            try:
                df = pd.read_csv(output_file, nrows=5)
                file_size_mb = output_file.stat().st_size / (1024 * 1024)
                print(f"✓ Success ({file_size_mb:.1f} MB, {len(pd.read_csv(output_file))} rows)")
                downloaded_files.append(output_file)
            except Exception as e:
                print(f"✗ Invalid CSV: {e}")
                output_file.unlink()  # Delete invalid file
                failed_downloads.append((year, "Invalid CSV"))

            # Be nice to the server
            time.sleep(0.5)

        except requests.exceptions.RequestException as e:
            print(f"✗ Failed: {e}")
            failed_downloads.append((year, str(e)))
            continue

    print("-" * 60)
    print(f"\n[Summary]")
    print(f"Successfully downloaded: {len(downloaded_files)} files")
    print(f"Failed downloads: {len(failed_downloads)}")

    if failed_downloads:
        print("\nFailed years:")
        for year, error in failed_downloads:
            print(f"  - {year}: {error}")

    return downloaded_files, failed_downloads


def inspect_data_structure(file_path):
    """
    Inspect the structure of a downloaded IESO file.

    Args:
        file_path: Path to CSV file
    """
    print(f"\n[Inspecting {file_path.name}]")
    df = pd.read_csv(file_path)

    print(f"Columns: {list(df.columns)}")
    print(f"Rows: {len(df)}")
    print(f"Date range: {df.iloc[0, 0]} to {df.iloc[-1, 0]}")
    print("\nFirst few rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())


def combine_all_years(input_dir='data/raw/ieso', output_file='data/raw/ieso_hourly_demand_2002_2025.csv'):
    """
    Combine all downloaded yearly files into a single CSV.

    Args:
        input_dir: Directory containing individual year files
        output_file: Path for combined output file
    """
    input_path = Path(input_dir)
    csv_files = sorted(input_path.glob('PUB_Demand_*.csv'))

    if not csv_files:
        print("No CSV files found to combine.")
        return None

    print(f"\n[Combining {len(csv_files)} files into single dataset]")

    all_data = []
    for file in csv_files:
        print(f"Reading {file.name}...", end=" ")
        try:
            df = pd.read_csv(file)
            all_data.append(df)
            print(f"✓ ({len(df)} rows)")
        except Exception as e:
            print(f"✗ Error: {e}")

    if not all_data:
        print("No data to combine.")
        return None

    # Concatenate all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)

    # Remove duplicates (if any overlap between years)
    original_rows = len(combined_df)
    combined_df = combined_df.drop_duplicates()
    duplicates_removed = original_rows - len(combined_df)

    # Sort by date
    if 'Date' in combined_df.columns:
        combined_df = combined_df.sort_values('Date').reset_index(drop=True)

    # Save combined file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)

    print(f"\n[Combined Dataset]")
    print(f"Total rows: {len(combined_df):,}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Date range: {combined_df.iloc[0, 0]} to {combined_df.iloc[-1, 0]}")
    print(f"Saved to: {output_path.absolute()}")

    return combined_df


if __name__ == "__main__":
    # Download all available data (2002-2025)
    downloaded_files, failed = download_ieso_demand_data(start_year=2002, end_year=2025)

    if downloaded_files:
        # Inspect one file to understand structure
        inspect_data_structure(downloaded_files[0])

        # Combine all years into single file
        combined_df = combine_all_years()

        print("\n" + "=" * 60)
        print("✓ IESO data download complete!")
        print("=" * 60)
    else:
        print("\n✗ No files were successfully downloaded.")
