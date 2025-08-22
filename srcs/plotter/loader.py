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
                    sample_features.append(0.0)
            features.append(sample_features)
    return np.array(features, dtype=np.float32), np.array(labels, dtype=str)
