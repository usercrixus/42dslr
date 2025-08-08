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
            for key in row:
                if key in cathegory:
                    val = row[key]
                    try:
                        sample_features.append(float(val))
                    except ValueError:
                        sample_features.append(0.0)
            features.append(sample_features)

    inputs = np.array(features, dtype=np.float32)
    return inputs, labels


def getInput(filename: str, cathegory: list):
    features = []

    with open(filename, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            sample = []
            for key in cathegory:
                try:
                    sample.append(float(row[key]))
                except:
                    sample.append(0.0)
            features.append(sample)

    return np.array(features, dtype=np.float32)
