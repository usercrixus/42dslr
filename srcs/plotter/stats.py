import math


def _clean(values):
    """Return a list of finite numbers only (exclude NaN/inf)."""
    out = []
    for v in values:
        # Exclude NaN and infinities
        if isinstance(v, (int, float)) and math.isfinite(v):
            out.append(float(v))
    return out


def stat_count(values):
    """Return the number of non-NaN values."""
    return len(_clean(values))


def stat_sum(values):
    """Return the sum of non-NaN values."""
    vals = _clean(values)
    total = 0.0
    for v in vals:
        total += v
    return total


def stat_mean(values):
    """Return the mean of non-NaN values."""
    vals = _clean(values)
    n = len(vals)
    if n == 0:
        return float("nan")
    return stat_sum(vals) / n


def stat_min(values):
    """Return the minimum (ignoring NaN)."""
    vals = _clean(values)
    if not vals:
        return float("nan")
    minimum = vals[0]
    for v in vals[1:]:
        if v < minimum:
            minimum = v
    return minimum


def stat_max(values):
    """Return the maximum (ignoring NaN)."""
    vals = _clean(values)
    if not vals:
        return float("nan")
    maximum = vals[0]
    for v in vals[1:]:
        if v > maximum:
            maximum = v
    return maximum


def stat_variance(values):
    """Return the population variance (divide by n), ignoring NaN."""
    vals = _clean(values)
    n = len(vals)
    if n == 0:
        return float("nan")
    mean = stat_mean(vals)
    s = 0.0
    for v in vals:
        s += (v - mean) ** 2
    return s / n


def stat_std(values):
    """Return the population standard deviation, ignoring NaN."""
    return math.sqrt(stat_variance(values))


def stat_percentile(values, percentile):
    """Return the given percentile (0-100) using nearest-rank, ignoring NaN."""
    vals = _clean(values)
    n = len(vals)
    if n == 0:
        return float("nan")
    sorted_vals = sorted(vals)
    pos = math.ceil(percentile / 100 * n) - 1
    pos = max(0, min(pos, n - 1))
    return sorted_vals[pos]


def stat_median(values):
    """Return the median (ignoring NaN)."""
    return stat_percentile(values, 50)


def stat_iqr(values):
    """Interquartile range: Q3 - Q1, ignoring NaN."""
    return stat_percentile(values, 75) - stat_percentile(values, 25)


def stat_range(values):
    """Range: max - min, ignoring NaN."""
    return stat_max(values) - stat_min(values)


def stat_cv(values):
    """Coefficient of variation = std / mean, ignoring NaN."""
    m = stat_mean(values)
    return float("nan") if m == 0 else stat_std(values) / m


def stat_mad(values):
    """Median absolute deviation, ignoring NaN."""
    vals = _clean(values)
    med = stat_median(vals)
    return stat_median([abs(v - med) for v in vals])


# Mapping
STAT_FUNCS = {
    "count": stat_count,
    "mean": stat_mean,
    "std": stat_std,
    "min": stat_min,
    "25%": lambda v: stat_percentile(v, 25),
    "50%": stat_median,
    "75%": lambda v: stat_percentile(v, 75),
    "max": stat_max,
    "iqr": stat_iqr,
    "range": stat_range,
    "cv": stat_cv,
    "mad": stat_mad,
}
