import sys
import pandas as pd
from stats import STAT_FUNCS
from loader import getInputAndLabel
from params import validCathegory

def format_statistics(df):
    df = df.drop(columns=["Hogwarts House"], errors="ignore")
    stats = {}
    for key, func in STAT_FUNCS.items():
        stats[key] = []
        for col in df.columns:
            stats[key].append(float(func(df[col].tolist())))
    out = pd.DataFrame(stats, index=df.columns)
    print(out.to_string(float_format="{:.6f}".format))


def main():
    if len(sys.argv) != 2:
        return
    feature_matrix, house_labels = getInputAndLabel(sys.argv[1], validCathegory)
    data = pd.DataFrame(feature_matrix, columns=validCathegory)
    format_statistics(data)


if __name__ == "__main__":
    main()
