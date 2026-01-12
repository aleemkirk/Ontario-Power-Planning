"""
Data loading utilities.

Loads and validates:
- Plant parameters (costs, capacity factors, emissions, etc.)
- Demand forecasts
- Initial capacity data
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple


class DataLoader:
    """Data loader for Ontario power system optimization."""

    def __init__(self, data_dir: str = 'data/processed/'):
        """
        Initialize data loader.

        Args:
            data_dir: Directory containing processed data files
        """
        self.data_dir = Path(data_dir)

    def load_plant_parameters(self) -> Dict:
        """
        Load plant parameters from JSON file.

        Returns:
            Dictionary with plant parameters:
            - capex: Capital costs ($/kW)
            - opex: Operating costs ($/MWh)
            - maintenance: Maintenance costs ($/MW/year)
            - emissions: Emission factors (tons CO2/MWh)
            - capacity_factor: Capacity factors (0-1)
            - ramp_rate: Ramp rates (MW/min per MW)
            - lead_time: Construction lead times (years)
            - lifespan: Plant lifespans (years)
        """
        file_path = self.data_dir / 'plant_parameters.json'

        if not file_path.exists():
            raise FileNotFoundError(f"Plant parameters file not found: {file_path}")

        with open(file_path, 'r') as f:
            data = json.load(f)

        # Validate required fields
        required_fields = ['capex', 'opex', 'maintenance', 'emissions',
                          'capacity_factor', 'ramp_rate', 'lead_time', 'lifespan']

        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        return data

    def load_demand_forecast(self) -> pd.DataFrame:
        """
        Load demand forecast from CSV file.

        Returns:
            DataFrame with columns:
            - year: Year
            - annual_demand: Annual demand (GWh)
            - peak_demand: Peak demand (MW)
        """
        file_path = self.data_dir / 'demand_forecast.csv'

        if not file_path.exists():
            raise FileNotFoundError(f"Demand forecast file not found: {file_path}")

        df = pd.read_csv(file_path)

        # Validate required columns
        required_cols = ['year', 'annual_demand', 'peak_demand']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        return df

    def load_initial_capacity(self) -> Dict[str, float]:
        """
        Load initial (2025) capacity data.

        Returns:
            Dictionary mapping plant types to initial capacity (MW)
        """
        file_path = self.data_dir / 'initial_capacity.json'

        if not file_path.exists():
            raise FileNotFoundError(f"Initial capacity file not found: {file_path}")

        with open(file_path, 'r') as f:
            data = json.load(f)

        return data

    def load_all_data(self) -> Tuple[Dict, pd.DataFrame, Dict]:
        """
        Load all data files.

        Returns:
            Tuple of (plant_parameters, demand_forecast, initial_capacity)
        """
        plant_params = self.load_plant_parameters()
        demand = self.load_demand_forecast()
        initial_cap = self.load_initial_capacity()

        return plant_params, demand, initial_cap

    def validate_data(self) -> bool:
        """
        Validate that all data is consistent and complete.

        Returns:
            True if validation passes
        """
        try:
            plant_params, demand, initial_cap = self.load_all_data()

            # Check plant types are consistent
            plant_types = set(plant_params['capex'].keys())
            assert plant_types == set(initial_cap.keys()), \
                "Plant types mismatch between parameters and initial capacity"

            # Check demand years are sequential
            years = demand['year'].values
            assert all(years[i+1] - years[i] == 1 for i in range(len(years)-1)), \
                "Demand forecast years are not sequential"

            # Check no negative values
            assert all(demand['annual_demand'] > 0), "Negative demand values found"
            assert all(demand['peak_demand'] > 0), "Negative peak demand values found"

            print("✓ Data validation passed")
            return True

        except Exception as e:
            print(f"✗ Data validation failed: {e}")
            return False


def create_sample_data():
    """
    Create sample data files for testing.

    This function creates the three required data files with sample data.
    """
    # Plant parameters
    plant_params = {
        'capex': {
            'nuclear': 17500,
            'wind': 1900,
            'solar': 1300,
            'gas': 1500,
            'hydro': 3000,
            'biofuel': 3200
        },
        'opex': {
            'nuclear': 22,
            'wind': 12,
            'solar': 12,
            'gas': 55,
            'hydro': 10,
            'biofuel': 42
        },
        'maintenance': {
            'nuclear': 105000,
            'wind': 45000,
            'solar': 20000,
            'gas': 20000,
            'hydro': 37500,
            'biofuel': 70000
        },
        'emissions': {
            'nuclear': 0.012,
            'wind': 0.011,
            'solar': 0.048,
            'gas': 0.45,
            'hydro': 0.024,
            'biofuel': 0.23
        },
        'capacity_factor': {
            'nuclear': 0.90,
            'wind': 0.35,
            'solar': 0.15,
            'gas': 0.55,
            'hydro': 0.50,
            'biofuel': 0.80
        },
        'ramp_rate': {
            'nuclear': 0.02,
            'wind': 0.05,
            'solar': 0.10,
            'gas': 0.04,
            'hydro': 0.15,
            'biofuel': 0.01
        },
        'lead_time': {
            'nuclear': 7,
            'wind': 2,
            'solar': 2,
            'gas': 3,
            'hydro': 6,
            'biofuel': 3
        },
        'lifespan': {
            'nuclear': 60,
            'wind': 25,
            'solar': 30,
            'gas': 35,
            'hydro': 100,
            'biofuel': 35
        }
    }

    # Initial capacity
    initial_capacity = {
        'nuclear': 13000,
        'wind': 5575,
        'solar': 2669,
        'gas': 10500,
        'hydro': 8500,
        'biofuel': 205
    }

    # Demand forecast (2025-2045)
    years = list(range(2025, 2046))
    base_demand = 151000  # GWh
    growth_rate = 0.022
    base_peak = 24000  # MW

    demand_data = {
        'year': years,
        'annual_demand': [base_demand * (1 + growth_rate) ** i for i in range(len(years))],
        'peak_demand': [base_peak * (1 + growth_rate) ** i for i in range(len(years))]
    }

    # Write files
    Path('data/processed').mkdir(parents=True, exist_ok=True)

    with open('data/processed/plant_parameters.json', 'w') as f:
        json.dump(plant_params, f, indent=2)

    with open('data/processed/initial_capacity.json', 'w') as f:
        json.dump(initial_capacity, f, indent=2)

    pd.DataFrame(demand_data).to_csv('data/processed/demand_forecast.csv', index=False)

    print("✓ Sample data files created successfully")


if __name__ == '__main__':
    # Create sample data and validate
    create_sample_data()
    loader = DataLoader()
    loader.validate_data()
