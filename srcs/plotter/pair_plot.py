import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from loader import getInputAndLabel
from params import selectedCathegory
from sklearn.feature_selection import f_classif


def compute_anova_f_scores(data: pd.DataFrame, houses: pd.Series) -> pd.Series:
    X = data.to_numpy(dtype=float)
    y = houses.to_numpy()
    f_vals, p_vals = f_classif(X, y)
    return pd.Series(f_vals, index=data.columns).sort_values(ascending=False)


def make_house_colors(houses: pd.Series) -> tuple[dict, np.ndarray]:
    uniques = sorted(pd.unique(houses.dropna()))
    palette = (
        plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])
    )
    mapping = {h: palette[i % len(palette)] for i, h in enumerate(uniques)}
    color_array = houses.map(mapping).to_numpy()
    return mapping, color_array


def plot(data: pd.DataFrame, houses: pd.Series) -> str:
    mapping, color_array = make_house_colors(houses)
    axes = scatter_matrix(
        data,
        figsize=(16, 16),
        diagonal="hist",
        color=color_array,
        alpha=0.6,
        range_padding=0.05,
    )
    n = len(data.columns)
    for i in range(n):
        axes[-1, i].xaxis.label.set_rotation(45)
        axes[-1, i].set_xlabel(data.columns[i], rotation=45, ha="right")
        axes[i, 0].yaxis.label.set_rotation(0)
        axes[i, 0].set_ylabel(data.columns[i], rotation=0, ha="right")
    handles = [
        plt.Line2D(
            [], [], marker="o", linestyle="", color=color, label=house, markersize=8
        )
        for house, color in mapping.items()
    ]
    plt.figlegend(
        handles=handles,
        labels=[h.get_label() for h in handles],
        loc="upper right",
        ncol=1,
        title="House",
    )
    plt.suptitle("Scatter Plot Matrix of Features (colored by House)", y=0.995)
    plt.tight_layout(pad=2.5, w_pad=0.7, h_pad=0.7)
    plt.show()


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 pairplot.py <dataset.csv>")
        return
    feature_matrix, house_labels = getInputAndLabel(sys.argv[1], selectedCathegory)
    data = pd.DataFrame(feature_matrix, columns=selectedCathegory)
    houses = pd.Series(house_labels, name="Hogwarts House", dtype="string")
    plot(data, houses)
    f_scores = compute_anova_f_scores(data, houses)
    print("\nFeature ranking by ANOVA F-score (higher = separates houses better):")
    for feature, score in f_scores.items():
        print(f"  {feature:30s}  F = {score:10.4f}")
    top_k = 8
    suggested_features = f_scores.head(top_k).index.tolist()
    print(f"\nSuggested features to keep for logistic regression (top {top_k}):")
    for feat in suggested_features:
        print(f"  - {feat}")


if __name__ == "__main__":
    main()
