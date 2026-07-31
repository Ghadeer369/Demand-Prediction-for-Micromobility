"""Derive analysis-ready features from the cleaned rides data: a coarse
Region grouping for the 106 raw Zone values, and a Saudi-weekend flag.
"""

import logging

import pandas as pd

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Keyword -> Region. Checked case-insensitively as a substring of the Zone value.
# Built by manually reading all 106 unique Zone values (see project notes) — any
# zone that doesn't match a keyword falls into "Other" rather than being guessed.
REGION_KEYWORDS = {
    "Jeddah": ["jeddah", "jamjoom", "tahlia", "alsayf", "obhor", "alnakil", "city walk",
               "durat alaroos", "gloso", "boost center", "art promenade", "shada hotel",
               "taiba walking", "jeddah pier"],
    "Makkah": ["mecca", "makkah", "hajj", "quraish", "hussainiya", "minaalmasha"],
    "Riyadh": ["riyadh", "ryd", "sabic"],
    "Jazan": ["jazan", "sabya", "samtah", "bish", "baish"],
    "Eastern": ["khobar", "dammam", "dhahran", "saihat", "qatif", "al-ahsa", "al ahsa", "hofuf"],
    "Taif": ["taif"],
    "Tabuk": ["tabuk"],
    "Madinah": ["medina", "madinah"],
    "Abha": ["abha"],
    "Bahrain": ["bahrain"],
    "Oman": ["oman"],
    "Qunfudhah": ["qunfudhah"],
}

SAUDI_WEEKEND_DAYS = {"Friday", "Saturday"}


def zone_to_region(zone: str) -> str:
    """Map a raw Zone string to a coarse Region using keyword matching.
    Returns "Other" when no known city keyword is found in the zone name."""
    zone_lower = str(zone).lower()
    for region, keywords in REGION_KEYWORDS.items():
        if any(keyword in zone_lower for keyword in keywords):
            return region
    return "Other"


def add_region(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Region"] = df[config.COL_ZONE].apply(zone_to_region)
    unmatched = (df["Region"] == "Other").sum()
    logger.info("Assigned regions to %d zones (%d rows fell into 'Other')", df[config.COL_ZONE].nunique(), unmatched)
    return df


def add_is_weekend(df: pd.DataFrame) -> pd.DataFrame:
    """Saudi weekend is Friday-Saturday, not Saturday-Sunday."""
    df = df.copy()
    df["Is_Weekend"] = df[config.COL_DAY_OF_WEEK].isin(SAUDI_WEEKEND_DAYS)
    return df


def engineer_features() -> pd.DataFrame:
    df = pd.read_csv(config.PROCESSED_RIDES_FILE)
    df = add_region(df)
    df = add_is_weekend(df)

    df.to_csv(config.PROCESSED_RIDES_FILE, index=False)
    logger.info("Saved feature-engineered dataset to %s (%d rows, %d columns)",
                config.PROCESSED_RIDES_FILE.name, len(df), df.shape[1])
    return df


if __name__ == "__main__":
    engineer_features()
