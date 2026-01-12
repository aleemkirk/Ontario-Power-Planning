"""
Create representative days from hourly demand data using K-means clustering.

This script clusters 365 daily demand profiles into a smaller number of
representative days (typically 12-24), dramatically reducing the size of
the hourly optimization model while maintaining 95%+ accuracy.

Method:
1. Extract daily demand profiles (365 days × 24 hours)
2. Calculate features: mean, peak, time-of-peak, std, season
3. Apply K-means clustering
4. Select most representative day from each cluster
5. Weight by cluster size (number of days represented)
6. Validate clustering quality

Output: JSON file with representative days and weights
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from datetime import datetime
import seaborn as sns


def load_hourly_demand(year=2024, data_file='data/processed/ieso_hourly_demand_2002_2025.csv'):
    """
    Load hourly demand data for a specific year.

    Args:
        year: Year to extract
        data_file: Path to processed IESO data

    Returns:
        DataFrame with hourly demand for specified year
    """
    df = pd.read_csv(data_file, parse_dates=['DateTime'])
    df_year = df[df['DateTime'].dt.year == year].copy()

    if len(df_year) == 0:
        raise ValueError(f"No data found for year {year}")

    print(f"Loaded {len(df_year)} hours for year {year}")
    return df_year


def extract_daily_profiles(df_hourly):
    """
    Reshape hourly data into daily profiles (365 days × 24 hours).

    Args:
        df_hourly: DataFrame with hourly demand data

    Returns:
        numpy array of shape (n_days, 24)
    """
    df_hourly = df_hourly.sort_values('DateTime').copy()
    df_hourly['Date'] = df_hourly['DateTime'].dt.date

    # Group by date and ensure 24 hours per day
    daily_profiles = []
    dates = []

    for date, group in df_hourly.groupby('Date'):
        if len(group) == 24:
            # Ensure hours are ordered 1-24
            group_sorted = group.sort_values('Hour')
            profile = group_sorted['Ontario Demand'].values
            daily_profiles.append(profile)
            dates.append(date)
        else:
            print(f"Warning: Date {date} has {len(group)} hours (expected 24), skipping")

    profiles_array = np.array(daily_profiles)
    print(f"Extracted {len(daily_profiles)} complete daily profiles")

    return profiles_array, dates


def extract_features(daily_profiles, dates):
    """
    Extract features from daily profiles for clustering.

    Features:
    - Mean demand
    - Peak demand
    - Hour of peak (0-23)
    - Standard deviation
    - Season (0=winter, 1=spring, 2=summer, 3=fall)
    - Day of week (0=Monday, 6=Sunday)
    - Is weekend (0=weekday, 1=weekend)

    Args:
        daily_profiles: array of shape (n_days, 24)
        dates: list of dates

    Returns:
        DataFrame with features for each day
    """
    n_days = len(daily_profiles)

    features = {
        'date': dates,
        'mean_demand': np.mean(daily_profiles, axis=1),
        'peak_demand': np.max(daily_profiles, axis=1),
        'min_demand': np.min(daily_profiles, axis=1),
        'hour_of_peak': np.argmax(daily_profiles, axis=1),
        'std_demand': np.std(daily_profiles, axis=1),
        'daily_range': np.max(daily_profiles, axis=1) - np.min(daily_profiles, axis=1),
    }

    # Add temporal features
    dates_dt = [pd.Timestamp(d) for d in dates]
    features['month'] = [d.month for d in dates_dt]
    features['day_of_week'] = [d.dayofweek for d in dates_dt]
    features['is_weekend'] = [1 if d.dayofweek >= 5 else 0 for d in dates_dt]

    # Add season (0=winter, 1=spring, 2=summer, 3=fall)
    def get_season(month):
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Spring
        elif month in [6, 7, 8]:
            return 2  # Summer
        else:
            return 3  # Fall

    features['season'] = [get_season(m) for m in features['month']]

    # Add load factor (how efficiently capacity is used)
    features['load_factor'] = features['mean_demand'] / features['peak_demand']

    df_features = pd.DataFrame(features)

    print(f"\nFeatures extracted:")
    print(df_features.describe())

    return df_features


def perform_clustering(df_features, daily_profiles, n_clusters=12, random_state=42):
    """
    Perform K-means clustering on daily demand profiles.

    Args:
        df_features: DataFrame with features for each day
        daily_profiles: array of shape (n_days, 24)
        n_clusters: Number of clusters (representative days)
        random_state: Random seed for reproducibility

    Returns:
        cluster_labels: Cluster assignment for each day
        kmeans: Fitted KMeans model
    """
    # Select features for clustering
    feature_cols = [
        'mean_demand', 'peak_demand', 'std_demand',
        'hour_of_peak', 'daily_range', 'load_factor',
        'season', 'is_weekend'
    ]

    X = df_features[feature_cols].values

    # Standardize features (important for K-means)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"\n[K-means Clustering with {n_clusters} clusters]")

    # Perform K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=50, max_iter=500)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Calculate silhouette score (measure of cluster quality)
    silhouette = silhouette_score(X_scaled, cluster_labels)
    print(f"Silhouette score: {silhouette:.3f} (higher is better, range -1 to 1)")

    # Print cluster sizes
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print(f"\nCluster sizes:")
    for cluster_id, count in zip(unique, counts):
        print(f"  Cluster {cluster_id}: {count} days")

    return cluster_labels, kmeans, scaler


def select_representative_days(daily_profiles, dates, cluster_labels, n_clusters):
    """
    Select the most representative day from each cluster.

    Strategy: For each cluster, select the day closest to the cluster centroid
    (measured in 24-dimensional demand space).

    Args:
        daily_profiles: array of shape (n_days, 24)
        dates: list of dates
        cluster_labels: Cluster assignment for each day
        n_clusters: Number of clusters

    Returns:
        Dictionary with representative day info for each cluster
    """
    representative_days = {}

    for cluster_id in range(n_clusters):
        # Get all days in this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_profiles = daily_profiles[cluster_mask]
        cluster_dates = [dates[i] for i, mask in enumerate(cluster_mask) if mask]

        if len(cluster_profiles) == 0:
            print(f"Warning: Cluster {cluster_id} is empty, skipping")
            continue

        # Calculate cluster centroid (mean profile)
        centroid = cluster_profiles.mean(axis=0)

        # Find day closest to centroid (Euclidean distance)
        distances = np.linalg.norm(cluster_profiles - centroid, axis=1)
        representative_idx = np.argmin(distances)

        representative_profile = cluster_profiles[representative_idx]
        representative_date = cluster_dates[representative_idx]

        # Calculate statistics for this cluster
        cluster_weight = len(cluster_profiles)  # Number of days represented
        cluster_mean_demand = cluster_profiles.mean()
        cluster_peak_demand = cluster_profiles.max(axis=1).mean()

        representative_days[cluster_id] = {
            'cluster_id': int(cluster_id),
            'representative_date': str(representative_date),
            'weight': int(cluster_weight),
            'hourly_demand': representative_profile.tolist(),
            'mean_demand': float(cluster_mean_demand),
            'peak_demand': float(cluster_peak_demand),
            'cluster_size': int(len(cluster_profiles)),
        }

        print(f"\nCluster {cluster_id}:")
        print(f"  Representative date: {representative_date}")
        print(f"  Weight: {cluster_weight} days")
        print(f"  Mean demand: {cluster_mean_demand:.0f} MW")
        print(f"  Peak demand: {cluster_peak_demand:.0f} MW")

    return representative_days


def validate_clustering(daily_profiles, representative_days, dates):
    """
    Validate clustering quality by comparing reconstructed vs actual demand.

    Metrics:
    - Energy balance: Total energy from rep days vs actual
    - Peak preservation: Max rep day peak vs actual peak
    - R² score: Goodness of fit
    - RMSE: Root mean squared error

    Args:
        daily_profiles: array of shape (n_days, 24)
        representative_days: dict with representative day info
        dates: list of dates

    Returns:
        Dictionary with validation metrics
    """
    print("\n[Validation]")

    # Calculate actual totals
    actual_total_energy = daily_profiles.sum()
    actual_peak_demand = daily_profiles.max()
    actual_mean_demand = daily_profiles.mean()

    # Calculate reconstructed totals using representative days
    reconstructed_total_energy = sum(
        rep_day['weight'] * sum(rep_day['hourly_demand'])
        for rep_day in representative_days.values()
    )

    rep_day_peaks = [max(rd['hourly_demand']) for rd in representative_days.values()]
    reconstructed_peak_demand = max(rep_day_peaks)

    # Energy balance error
    energy_error = abs(reconstructed_total_energy - actual_total_energy) / actual_total_energy
    print(f"Energy balance:")
    print(f"  Actual total: {actual_total_energy:,.0f} MWh")
    print(f"  Reconstructed: {reconstructed_total_energy:,.0f} MWh")
    print(f"  Error: {energy_error*100:.2f}%")

    # Peak preservation
    peak_error = abs(reconstructed_peak_demand - actual_peak_demand) / actual_peak_demand
    print(f"\nPeak demand:")
    print(f"  Actual peak: {actual_peak_demand:.0f} MW")
    print(f"  Reconstructed peak: {reconstructed_peak_demand:.0f} MW")
    print(f"  Error: {peak_error*100:.2f}%")

    # Calculate R² (coefficient of determination)
    # Reconstruct daily demand using weighted average of rep days
    # This is approximate since we don't know which day maps to which cluster
    total_weight = sum(rd['weight'] for rd in representative_days.values())
    weighted_mean_profile = sum(
        (rd['weight'] / total_weight) * np.array(rd['hourly_demand'])
        for rd in representative_days.values()
    )

    # Calculate R² for mean daily profile
    mean_actual_profile = daily_profiles.mean(axis=0)
    ss_tot = np.sum((mean_actual_profile - mean_actual_profile.mean()) ** 2)
    ss_res = np.sum((mean_actual_profile - weighted_mean_profile) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    print(f"\nR² score (mean profile): {r_squared:.4f}")

    validation_metrics = {
        'actual_total_energy_MWh': float(actual_total_energy),
        'reconstructed_total_energy_MWh': float(reconstructed_total_energy),
        'energy_error_pct': float(energy_error * 100),
        'actual_peak_MW': float(actual_peak_demand),
        'reconstructed_peak_MW': float(reconstructed_peak_demand),
        'peak_error_pct': float(peak_error * 100),
        'r_squared': float(r_squared),
        'n_representative_days': len(representative_days),
        'total_days': len(daily_profiles),
    }

    # Check if validation passes
    if energy_error < 0.03 and peak_error < 0.05 and r_squared > 0.95:
        print(f"\n✓ Validation PASSED (energy error <3%, peak error <5%, R² >0.95)")
    else:
        print(f"\n⚠ Validation WARNING: May need more clusters or different features")

    return validation_metrics


def plot_representative_days(representative_days, output_path='results/figures/representative_days.png'):
    """
    Visualize all representative days in a grid.

    Args:
        representative_days: dict with representative day info
        output_path: Path to save figure
    """
    n_clusters = len(representative_days)
    n_cols = 3
    n_rows = (n_clusters + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3))
    axes = axes.flatten() if n_clusters > 1 else [axes]

    hours = list(range(24))

    for idx, (cluster_id, rep_day) in enumerate(sorted(representative_days.items())):
        ax = axes[idx]

        demand = rep_day['hourly_demand']
        weight = rep_day['weight']
        date = rep_day['representative_date']

        ax.plot(hours, demand, linewidth=2, marker='o', markersize=4)
        ax.set_title(f"Cluster {cluster_id}: {weight} days\n{date}", fontsize=10, fontweight='bold')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Demand (MW)')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 23)

    # Hide unused subplots
    for idx in range(n_clusters, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f'Representative Days ({n_clusters} clusters)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save figure
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved representative days plot to {output_file}")
    plt.close()


def save_representative_days(representative_days, validation_metrics, year, n_clusters,
                             output_path='data/processed/representative_days_12clusters.json'):
    """
    Save representative days to JSON file.

    Args:
        representative_days: dict with representative day info
        validation_metrics: dict with validation results
        year: Year used for template
        n_clusters: Number of clusters
        output_path: Path to save JSON file
    """
    output_data = {
        'metadata': {
            'n_clusters': n_clusters,
            'template_year': year,
            'total_weight': sum(rd['weight'] for rd in representative_days.values()),
            'creation_date': datetime.now().isoformat(),
        },
        'validation_metrics': validation_metrics,
        'representative_days': representative_days,
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"✓ Saved representative days to {output_file}")

    return output_file


def create_representative_days(year=2024, n_clusters=12,
                               data_file='data/processed/ieso_hourly_demand_2002_2025.csv'):
    """
    Main function to create representative days from hourly demand data.

    Args:
        year: Year to use as template
        n_clusters: Number of clusters (representative days)
        data_file: Path to processed IESO data

    Returns:
        Path to saved JSON file
    """
    print("=" * 60)
    print(f"Creating {n_clusters} Representative Days for {year}")
    print("=" * 60)

    # Step 1: Load data
    df_hourly = load_hourly_demand(year, data_file)

    # Step 2: Extract daily profiles
    daily_profiles, dates = extract_daily_profiles(df_hourly)

    # Step 3: Extract features
    df_features = extract_features(daily_profiles, dates)

    # Step 4: Perform clustering
    cluster_labels, kmeans, scaler = perform_clustering(df_features, daily_profiles, n_clusters)

    # Add cluster labels to features dataframe
    df_features['cluster'] = cluster_labels

    # Step 5: Select representative days
    representative_days = select_representative_days(daily_profiles, dates, cluster_labels, n_clusters)

    # Step 6: Validate clustering
    validation_metrics = validate_clustering(daily_profiles, representative_days, dates)

    # Step 7: Visualize
    plot_representative_days(representative_days)

    # Step 8: Save to file
    output_path = f'data/processed/representative_days_{n_clusters}clusters.json'
    output_file = save_representative_days(representative_days, validation_metrics, year, n_clusters, output_path)

    print("\n" + "=" * 60)
    print(f"✓ Representative days creation complete!")
    print("=" * 60)

    return output_file


if __name__ == "__main__":
    # Create 12-cluster representative days (standard)
    create_representative_days(year=2024, n_clusters=12)

    # Optionally create 24-cluster version (high fidelity)
    # create_representative_days(year=2024, n_clusters=24)
