import sys
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
    if len(sys.argv) != 2:
        return
    df = load_and_prepare(sys.argv[1])
    plots_histograms(df, "histograms")


if __name__ == '__main__':
    main()
