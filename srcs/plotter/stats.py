import math

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
    pos = math.ceil(percentile / 100 * n) - 1
    pos = max(0, min(pos, n-1))
    return sorted_vals[pos]

# Mapping
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
