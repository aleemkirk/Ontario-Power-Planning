"""System-wide constants for Ontario Power Planning optimization."""

# Time conversions
HOURS_PER_YEAR = 8760
MINUTES_PER_HOUR = 60
HOURS_PER_DAY = 24

# System parameters
DEFAULT_RESERVE_MARGIN = 0.15  # 15%
DEFAULT_DISCOUNT_RATE = 0.0392  # 3.92% real

# Unit conversions
MW_TO_KW = 1000
GW_TO_MW = 1000
MWH_TO_GWH = 0.001

# Model defaults
DEFAULT_N_REP_DAYS = 12
DEFAULT_RAMP_PENALTY = 1000.0
DEFAULT_TIME_LIMIT = 600  # seconds
