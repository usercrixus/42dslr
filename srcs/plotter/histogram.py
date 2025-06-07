import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
from loader import load_and_prepare


def plots_histograms(df: pd.DataFrame, output_dir: str):
    """
    Generate and save a histogram for each column in df.
    Histograms are saved under `output_dir`.
    """
    os.makedirs(output_dir, exist_ok=True)
    for feature in df.columns:
        data = df[feature]
        plt.figure(figsize=(8, 6))
        plt.hist(data, bins=20, edgecolor='black')
        plt.title(f"Histogram of {feature}")
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.grid(True)

        filename = os.path.join(
            output_dir,
            f"hist_{feature.replace(' ', '_')}.png"
        )
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved histogram: {filename}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate histograms for each numeric feature in the CSV'
    )
    parser.add_argument('csv_file', help='Path to the CSV file')
    args = parser.parse_args()

    # Load and prepare the DataFrame
    df = load_and_prepare(args.csv_file)
    # Drop unwanted metadata columns
    df = df.drop(columns=[
        'Index',
        'First Name',
        'Last Name',
        'Best Hand'
    ], errors='ignore')

    # Generate and save histograms
    plots_histograms(df, "histograms")


if __name__ == '__main__':
    main()
