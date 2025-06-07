#!/usr/bin/env python3
import argparse
import math
import pandas as pd

# --- Statistic functions implemented from scratch ---
def stat_count(values):
    """Return the number of values."""
    count = 0
    for _ in values:
        count += 1
    return count


def stat_sum(values):
    """Return the sum of the values."""
    total = 0.0
    for v in values:
        total += v
    return total


def stat_mean(values):
    """Return the mean of the values."""
    n = stat_count(values)
    if n == 0:
        return float('nan')
    return stat_sum(values) / n


def stat_min(values):
    """Return the minimum value."""
    if not values:
        return float('nan')
    minimum = values[0]
    for v in values[1:]:
        if v < minimum:
            minimum = v
    return minimum


def stat_max(values):
    """Return the maximum value."""
    if not values:
        return float('nan')
    maximum = values[0]
    for v in values[1:]:
        if v > maximum:
            maximum = v
    return maximum


def stat_variance(values):
    """Return the population variance (divide by n)."""
    n = stat_count(values)
    if n == 0:
        return float('nan')
    mean = stat_mean(values)
    s = 0.0
    for v in values:
        s += (v - mean) ** 2
    return s / n


def stat_std(values):
    """Return the population standard deviation."""
    return math.sqrt(stat_variance(values))


def stat_percentile(values, percentile):
    """Return the given percentile (0-100) using nearest-rank method."""
    n = stat_count(values)
    if n == 0:
        return float('nan')
    sorted_vals = sorted(values)
    # nearest-rank position (1-based), then convert to 0-based
    pos = math.ceil(percentile / 100 * n) - 1
    pos = max(0, min(pos, n-1))
    return sorted_vals[pos]

# --- Columns to read ---
SELECTED_FEATURES = [
    'Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts',
    'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic',
    'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying',
    'Birthday', 'Best Hand'
]

# --- Load and preprocess CSV with pandas ---
def load_and_prepare(path):
    """
    Read the CSV using pandas, keep only SELECTED_FEATURES,
    convert Birthday to epoch seconds, and Best Hand to numeric.
    """
    df = pd.read_csv(path, usecols=SELECTED_FEATURES)
    # Convert Birthday to seconds since epoch
    df['Birthday'] = (pd.to_datetime(df['Birthday'], errors='coerce').astype('int64') // 10**9)
    # Map Best Hand to numeric
    df['Best Hand'] = df['Best Hand'].map({'Left': 1.0, 'Right': 0.0})
    # Fill any missing entries with 0 to ensure equal length
    df = df.fillna(0)
    return df

# --- Mapping statistic names to functions ---
STAT_FUNCS = {
    'count': stat_count,
    'mean': stat_mean,
    'std': stat_std,
    'min': stat_min,
    '25%': lambda v: stat_percentile(v, 25),
    '50%': lambda v: stat_percentile(v, 50),
    '75%': lambda v: stat_percentile(v, 75),
    'max': stat_max,
}

# --- Format and print the table ---
def format_statistics(df):
    """Compute and print descriptive statistics for each column."""
    STATS = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    # Build a DataFrame by applying our custom functions directly into rows
    # Build a dict of raw numbers by applying our custom functions directly
    # Use full lists (including zeros) so counts are equal
    stats_dict = {
        stat: [STAT_FUNCS[stat](df[col].tolist()) for col in df.columns] for stat in STATS
    }
    # Create DataFrame with features as rows and stats as columns
    stats_df = pd.DataFrame(stats_dict, index=df.columns)
    # Replace any missing (NaN) values with 0 to maintain a perfect matrix
    stats_df = stats_df.fillna(0)
    # Print with six-decimal formatting
    print(stats_df.to_string(float_format="{:.6f}".format))

# --- Main ---
def main():
    parser = argparse.ArgumentParser(
        description='Compute descriptive statistics for selected CSV fields.'
    )
    parser.add_argument('csv_file', help='Path to the CSV file')
    args = parser.parse_args()

    df = load_and_prepare(args.csv_file)
    format_statistics(df)

if __name__ == '__main__':
    main()
