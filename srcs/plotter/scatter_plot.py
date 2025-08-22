# scatter.py
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from loader import getInputAndLabel
from params import selectedCathegory


def find_most_similar_features_by_correlation(
    dataframe: pd.DataFrame,
) -> tuple[str, str, float]:
    """
    Return the pair of feature names with the highest absolute Pearson correlation,
    along with the signed correlation value.
    """
    correlation = dataframe.corr(method="pearson")
    abs_correlation = correlation.abs()

    # Ignore self-correlation on the diagonal
    np.fill_diagonal(abs_correlation.values, np.nan)

    # Find the location (row, col) of the maximum absolute correlation
    max_location = np.unravel_index(
        np.nanargmax(abs_correlation.values), abs_correlation.shape
    )
    feature_x = abs_correlation.index[max_location[0]]
    feature_y = abs_correlation.columns[max_location[1]]

    signed_corr_value = correlation.loc[feature_x, feature_y]
    return feature_x, feature_y, float(signed_corr_value)


def plot_scatter_for_features(
    dataframe: pd.DataFrame, feature_x: str, feature_y: str, corr_value: float
) -> None:
    """
    Display a single scatter plot for feature_x vs feature_y.
    """
    plt.figure(figsize=(9, 6))
    plt.scatter(
        dataframe[feature_x].to_numpy(),
        dataframe[feature_y].to_numpy(),
        alpha=0.6,
        edgecolor="none",
    )
    plt.title(
        f"Most similar features: {feature_x} vs {feature_y} (corr = {corr_value:.4f})"
    )
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 scatter.py <dataset.csv>")
        return
    # Load numeric features and house labels using your project loader utilities
    feature_matrix, _house_labels = getInputAndLabel(sys.argv[1], selectedCathegory)
    # Wrap features with column names for clarity and correlation by name
    dataframe = pd.DataFrame(feature_matrix, columns=selectedCathegory)
    # 1) Find the most similar pair (highest absolute correlation)
    feature_x, feature_y, corr_value = find_most_similar_features_by_correlation(
        dataframe
    )
    print(
        f"Most similar features: {feature_x} and {feature_y} (corr = {corr_value:.6f})"
    )
    # 2) Plot exactly one scatter answering the question
    plot_scatter_for_features(dataframe, feature_x, feature_y, corr_value)


if __name__ == "__main__":
    main()
