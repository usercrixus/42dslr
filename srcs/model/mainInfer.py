import sys
import csv
import numpy as np
from LogisticRegression import LogisticRegression

def load_features_only(filename: str):
    features = []

    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            sample = []
            for key in ["Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts",
                        "Divination", "Muggle Studies", "Ancient Runes", "History of Magic",
                        "Transfiguration", "Potions", "Care of Magical Creatures", "Charms", "Flying"]:
                try:
                    sample.append(float(row[key]))
                except:
                    sample.append(0.0)
            features.append(sample)

    return np.array(features, dtype=np.float32)  # (num_samples, input_dim)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 mainInfer.py <dataset.csv>")
        sys.exit(1)

    inputPath = sys.argv[1]

    # Load model parameters
    weights = np.load("model_weights.npy")
    biais = np.load("model_bias.npy")
    labels = np.load("model_labels.npy").tolist()
    inputMean = np.load("X_mean.npy")
    inputStd = np.load("X_std.npy")

    # Load and normalize data
    input = load_features_only(inputPath)
    input = (input - inputMean) / (inputStd + 1e-8)

    # Build dummy model and inject weights
    model = LogisticRegression(inputDim=input.shape[1], labels=labels)
    model.layer.weights = weights
    model.layer.biais = biais

    predictions = model.predict(input)
    print("Index,Hogwarts House")
    for i, pred in enumerate(predictions):
        print(f"{i},{pred}")
