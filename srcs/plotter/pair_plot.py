import argparse
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
    parser = argparse.ArgumentParser(
        description='Generate a clean scatter matrix for selected features'
    )
    parser.add_argument('csv_file', help='Path to CSV file')
    args = parser.parse_args()

    # Load and prepare data
    df = load_and_prepare(args.csv_file)
    # Drop unwanted metadata columns
    df = df.drop(columns=[
        'Index',
        'First Name',
        'Last Name',
        'Best Hand'
    ], errors='ignore')

    # Prepare data for pair plot

    # Plot and save
    plot_pair_matrix(df, "pair_plots")


if __name__ == '__main__':
    main()
