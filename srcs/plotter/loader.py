import csv
import numpy as np


def getInputAndLabel(filename: str, cathegory: list):
    features = []
    labels = []

    with open(filename, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            label = row["Hogwarts House"]
            labels.append(label)

            sample_features = []
            for key in cathegory:
                try:
                    sample_features.append(float(row[key]))
                except ValueError:
                    # Preserve missing/invalid numeric values as NaN for later handling
                    sample_features.append(float("nan"))
            features.append(sample_features)
    return np.array(features, dtype=np.float32), np.array(labels, dtype=str)
