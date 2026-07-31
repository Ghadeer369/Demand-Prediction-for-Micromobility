"""Central configuration: paths, constants, and column names used across the project."""

from pathlib import Path

# Project root = the folder that contains this src/ package
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data paths
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RAW_RIDES_FILE = RAW_DATA_DIR / "RidesDetails.csv"
PROCESSED_RIDES_FILE = PROCESSED_DATA_DIR / "rides_clean.csv"

# Model & output paths
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"

# Reproducibility
RANDOM_SEED = 42

# Spiders Mobility brand colors, used across every saved chart so figures,
# the README, and the Tableau dashboard all look like one consistent product.
BRAND_PRIMARY = "#4CAF7D"   # mint green (Spiders scooters/logo)
BRAND_ACCENT = "#7B85C9"    # soft lavender-blue (from the presentation gradient)
BRAND_TEXT = "#2D2D2D"      # charcoal

# All forecasting models are evaluated on the SAME held-out window: the last
# TEST_DAYS days, chronologically. 30 days is also a realistic operational
# forecast horizon (a 130+ day blind forecast is not something a real
# demand-planning team would ever ask a model for).
TEST_DAYS = 30

# Raw CSV format (confirmed from the actual RidesDetails.csv file)
RAW_CSV_SEPARATOR = ";"
RAW_CSV_ENCODING = "cp1256"  # Windows Arabic encoding — plain "utf-8" will crash on Customer names

# Raw dataset columns (exactly as they appear in RidesDetails.csv — never invent columns)
COL_RIDE_ID = "RideId"
COL_RIDE_START_TIME = "Ride_Start_Time"  # holds BOTH date and time, e.g. "03/11/21 21:22"
COL_RIDE_END_TIME = "Ride_End_Time"      # same format
COL_ZONE = "Zone"
COL_CUSTOMER = "Customer"
COL_TOTAL_TIME = "Total_Time"

# Engineered columns — these do NOT exist in the raw file.
# We will derive them ourselves in preprocessing.py / feature_engineering.py.
COL_RIDE_START_DATE = "Ride_Start_Date"  # derived: date part of Ride_Start_Time
COL_DAY_OF_WEEK = "Day_of_Week"          # derived: weekday name from Ride_Start_Date
COL_TIMEZONE = "TimeZone"                # derived: Morning/Afternoon/Evening/Night bucket from the hour
