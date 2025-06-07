import argparse
import pandas as pd
from stats import STAT_FUNCS
from loader import load_and_prepare

def format_statistics(df):
    # Drop the label column
    df = df.drop(columns=['Hogwarts House'], errors='ignore')
    stats_order = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    stats = {}
    # Build a dict of stats: each key is a stat name, value is list over columns
    for stat in stats_order:
        stats[stat] = []
        for col in df.columns:
            # ensure float for consistent formatting
            stats[stat].append(float(STAT_FUNCS[stat](df[col].tolist())))
    # Create DataFrame: rows=features, cols=stats
    out = pd.DataFrame(stats, index=df.columns)
    # Print with six-decimal formatting
    print(out.to_string(float_format="{:.6f}".format))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_file', help='Path to CSV file')
    args = parser.parse_args()
    df = load_and_prepare(args.csv_file)

    # Drop unwanted columns if they exist
    df = df.drop(columns=[
        'Index',
        'First Name',
        'Last Name',
        'Best Hand'
    ], errors='ignore')

    format_statistics(df)


if __name__=='__main__':
    main()