import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
from loader import load_and_prepare


def find_most_similar_features(df: pd.DataFrame) -> tuple[str, str, float]:
    """
    Compute the correlation matrix of numeric features in df,
    ignore self-correlations, and return the pair of features
    with the highest absolute correlation along with the correlation value.
    """
    # Exclude Hogwarts House from feature correlations if present
    feat_df = df.drop(columns=["Hogwarts House"], errors="ignore")

    corr = feat_df.corr()
    corr_abs = corr.abs()
    # Zero out diagonal
    for col in corr_abs.columns:
        corr_abs.loc[col, col] = 0

    feat_x, feat_y = corr_abs.unstack().idxmax()
    corr_val = corr.loc[feat_x, feat_y]
    return feat_x, feat_y, corr_val


def plot_scatter(df: pd.DataFrame, feat_x: str, feat_y: str, output_dir: str) -> str:
    """
    Generate and save a basic scatter plot for feat_x vs feat_y in df.
    Returns the filepath of the saved image.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"scatter_{feat_x.replace(' ', '_')}_{feat_y.replace(' ', '_')}.png"
    filepath = os.path.join(output_dir, filename)

    plt.figure(figsize=(8, 6))
    plt.scatter(df[feat_x], df[feat_y], alpha=0.6)
    plt.title(f"Scatter: {feat_x} vs {feat_y}")
    plt.xlabel(feat_x)
    plt.ylabel(feat_y)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    return filepath


def main():
    if len(sys.argv) != 2:
        return
    df = load_and_prepare(sys.argv[1])

    # Find most correlated features
    feat_x, feat_y, corr_val = find_most_similar_features(df)
    print(f"Most similar features: {feat_x} and {feat_y} (corr={corr_val:.6f})")

    # Plot and save
    saved_path = plot_scatter(df, feat_x, feat_y, "scatter")
    print(f"Saved scatter plot to {saved_path}")


if __name__ == "__main__":
    main()
