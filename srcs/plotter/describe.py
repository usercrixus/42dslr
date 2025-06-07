#!/usr/bin/env python3
import argparse
import csv
import math
import datetime

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
    """Return the sample variance (ddof=1)."""
    if not values:
        return float('nan')
    n = stat_count(values)
    mean = stat_mean(values)
    s = 0.0
    for v in values:
        s += (v - mean) ** 2
    return s / n


def stat_std(values):
    """Return the sample standard deviation."""
    return math.sqrt(stat_variance(values))


def stat_percentile(values, percentile):
    """Return the given percentile (0-100) using interpolation."""
    n = stat_count(values)
    if n == 0:
        return float('nan')
    sorted_vals = sorted(values)
    rank = math.ceil(percentile / 100 * n)
    if rank >= n:
        return sorted_vals[-1]
    return sorted_vals[rank]

# Mapping of statistic names to functions
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

def parse_birthday(val):
    """Convert ISO date string to seconds since epoch."""
    try:
        dt = datetime.datetime.fromisoformat(val)
        return int(dt.replace(tzinfo=datetime.timezone.utc).timestamp())
    except Exception:
        return None


def parse_best_hand(val):
    """Map 'Left'->1, 'Right'->0."""
    if val.lower() == 'left':
        return 1.0
    if val.lower() == 'right':
        return 0.0
    return None

def load_selected_numeric_csv(path):
    """Load CSV, drop 'Index', keep only selected features with numeric parsing."""
    # --- CSV loading and selecting only specified features ---
    SELECTED_FEATURES = [
        'Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts',
        'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic',
        'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying',
        'Birthday', 'Best Hand'
    ]
    with open(path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        columns = [col for col in header if col in SELECTED_FEATURES]
        data = {col: [] for col in columns}
        for row in reader:
            for col, val in zip(header, row):
                if col in columns:
                    num = None
                    if col == 'Birthday':
                        num = parse_birthday(val)
                    elif col == 'Best Hand':
                        num = parse_best_hand(val)
                    else:
                        try:
                            num = float(val)
                        except ValueError:
                            num = 0
                    data[col].append(num)
    return columns, data

# --- Table formatting with aligned decimal points ---
def format_statistics(columns, data):
    """Compute and format stats for each numeric column with decimal alignment."""
    stats_order = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    # Compute stat values
    values = {stat: [STAT_FUNCS[stat](data[col]) for col in columns] for stat in stats_order}
    # Compute widths for integer part and total width per column
    int_width = {}
    for col in columns:
        # Determine max integer-width before decimal
        max_int = 0
        for stat in stats_order:
            s = f"{values[stat][columns.index(col)]:.6f}"
            int_part = s.split('.')[0]
            if len(int_part) > max_int:
                max_int = len(int_part)
        int_width[col] = max_int
    # Total column width: int + '.' + 6 decimals or header length
    col_width = {}
    for col in columns:
        header_len = len(col)
        total = int_width[col] + 1 + 6
        col_width[col] = max(header_len, total)
    # Label column width
    label_width = max(len(s) for s in stats_order)
    # Build lines
    lines = []
    # Header line
    header_line = ' ' * (label_width + 1) + ' ' + ' '.join(col.rjust(col_width[col]) for col in columns)
    lines.append(header_line)
    # Stat rows
    for stat in stats_order:
        row_cells = []
        for col in columns:
            val = values[stat][columns.index(col)]
            s = f"{val:.6f}"
            ip, fp = s.split('.')
            ip = ip.rjust(int_width[col])
            cell = ip + '.' + fp
            cell = cell.rjust(col_width[col])
            row_cells.append(cell)
        row = stat.rjust(label_width) + ' ' + ' '.join(row_cells)
        lines.append(row)
    return '\n'.join(lines)

# --- Main entry point ---
def main():
    parser = argparse.ArgumentParser( description='Compute descriptive statistics for specified CSV fields without built-ins.')
    parser.add_argument('csv_file', help='Path to the CSV file')
    args = parser.parse_args()

    columns, data = load_selected_numeric_csv(args.csv_file)
    table = format_statistics(columns, data)
    print(table)

if __name__ == '__main__':
    main()
