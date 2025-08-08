import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
from loader import load_and_prepare

def plot_pair_matrix(feature_df: pd.DataFrame, output_dir: str):
    """
    Generate and save a scatter matrix (pair plot) for the given features.
    Saves a single PNG named 'pair_plot.png' inside output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 10))
    scatter_matrix(
        feature_df,
        alpha=0.5,
        diagonal='hist',
        figsize=(10, 10),
        hist_kwds={'bins': 15},
        marker='.'
    )
    plt.suptitle('Pair Plot of Selected Features', y=0.92)

    filepath = os.path.join(output_dir, 'pair_plot.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved pair plot to {filepath}")


def main():
    if len(sys.argv) != 2:
        return
    df = load_and_prepare(sys.argv[1])
    plot_pair_matrix(df, "pair_plots")


if __name__ == '__main__':
    main()
