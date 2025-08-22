# pairplot.py
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from loader import getInputAndLabel
from params import selectedCathegory


def make_house_colors(house_series: pd.Series) -> tuple[dict, np.ndarray]:
    """
    Assign a distinct matplotlib color to each house and return:
      - mapping: {house_name -> color}
      - color_array: color per sample (aligned with house_series)
    """
    unique_houses = sorted(house_series.unique().tolist())
    available_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])
    if len(available_colors) < len(unique_houses):
        # extend with default color cycle if needed
        repeats = (len(unique_houses) // len(available_colors)) + 1
        available_colors = (available_colors * repeats)[: len(unique_houses)]

    mapping = {house: available_colors[i] for i, house in enumerate(unique_houses)}
    color_array = house_series.map(mapping).to_numpy()
    return mapping, color_array


def compute_anova_f_scores(df_features: pd.DataFrame, houses: pd.Series) -> pd.Series:
    """
    For each feature column:
      - Compute ANOVA-like F-score across 4 houses:
            F = MS_between / MS_within
        where:
            MS_between = SS_between / (k - 1)
            MS_within  = SS_within  / (N - k)
    Higher F => stronger separation of houses => more promising for logistic regression.
    """
    k = houses.nunique()
    N = len(houses)
    if k < 2:
        raise ValueError("Need at least two houses to compute ANOVA scores.")
    if N != len(df_features):
        raise ValueError("Length mismatch between features and houses.")

    f_scores = {}
    for feature in df_features.columns:
        x = df_features[feature].to_numpy()
        # Drop NaNs jointly with houses to keep alignment (simple approach)
        mask = ~np.isnan(x) & houses.notna().to_numpy()
        x = x[mask]
        h = houses.to_numpy()[mask]

        if x.size == 0:
            f_scores[feature] = 0.0
            continue

        overall_mean = x.mean()

        # Compute group stats
        ss_between = 0.0
        ss_within = 0.0
        groups = 0
        for house in np.unique(h):
            group_values = x[h == house]
            n_g = group_values.size
            if n_g < 2:
                # With one point, it contributes to between but not within variance
                group_mean = group_values.mean()
                ss_between += n_g * (group_mean - overall_mean) ** 2
                groups += 1
                continue
            group_mean = group_values.mean()
            group_var = group_values.var(ddof=1)  # sample variance
            ss_between += n_g * (group_mean - overall_mean) ** 2
            ss_within += (n_g - 1) * group_var
            groups += 1

        # Use observed number of groups (could be < k if some houses fully NaN)
        dof_between = max(1, groups - 1)
        dof_within = max(1, x.size - groups)

        ms_between = ss_between / dof_between
        ms_within = ss_within / dof_within if ss_within > 0 else np.inf  # avoid division by zero

        f_value = ms_between / ms_within if ms_within != 0 and not np.isinf(ms_within) else np.inf
        f_scores[feature] = float(f_value)

    return pd.Series(f_scores).sort_values(ascending=False)


def plot_pair_matrix(df_features: pd.DataFrame, houses: pd.Series, output_dir: str = "pairplot") -> str:
    """
    Draw and save a scatter-plot matrix of the selected features, colored by house.
    Returns the path to the saved PNG.
    """
    os.makedirs(output_dir, exist_ok=True)

    house_to_color, sample_colors = make_house_colors(houses)
    axes = scatter_matrix(
        df_features,
        figsize=(16, 16),   # was (12, 12) → larger overall
        diagonal="hist",
        color=sample_colors,
        alpha=0.6,
        range_padding=0.05,
    )

    # Rotate bottom x-labels and left y-labels for readability
    n = len(df_features.columns)
    for i in range(n):
        # bottom row (x labels)
        axes[-1, i].xaxis.label.set_rotation(45)
        axes[-1, i].set_xlabel(df_features.columns[i], rotation=45, ha="right")

        # left column (y labels)
        axes[i, 0].yaxis.label.set_rotation(0)
        axes[i, 0].set_ylabel(df_features.columns[i], rotation=0, ha="right", va="center")

    # Legend (one entry per house)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=color, label=house, markersize=8)
        for house, color in house_to_color.items()
    ]
    plt.figlegend(handles=handles, labels=[h.get_label() for h in handles], loc="upper right", ncol=1, title="House")

    outpath = os.path.join(output_dir, "pairplot.png")
    plt.suptitle("Scatter Plot Matrix of Features (colored by House)", y=0.995)

    # Use tight_layout with padding to avoid crowding
    plt.tight_layout(pad=2.5, w_pad=0.7, h_pad=0.7)

    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.show()
    return outpath



def main():
    if len(sys.argv) != 2:
        print("Usage: python3 pairplot.py <dataset.csv>")
        return

    # Load numeric features (columns = selectedCathegory) and house labels
    feature_matrix, house_labels = getInputAndLabel(sys.argv[1], selectedCathegory)
    df_features = pd.DataFrame(feature_matrix, columns=selectedCathegory)
    houses = pd.Series(house_labels, name="Hogwarts House", dtype="string")

    # 1) Visual exploration: scatter matrix
    saved_path = plot_pair_matrix(df_features, houses)
    print(f"Saved pair plot to: {saved_path}")

    # 2) Feature ranking by ANOVA F-score
    f_scores = compute_anova_f_scores(df_features, houses)
    print("\nFeature ranking by ANOVA F-score (higher = separates houses better):")
    for feature, score in f_scores.items():
        print(f"  {feature:30s}  F = {score:10.4f}")

    # 3) Suggested selection of features
    top_k = 8   # choose how many features you want to keep
    suggested_features = f_scores.head(top_k).index.tolist()

    print(f"\nSuggested features to keep for logistic regression (top {top_k}):")
    for feat in suggested_features:
        print(f"  - {feat}")



if __name__ == "__main__":
    main()
