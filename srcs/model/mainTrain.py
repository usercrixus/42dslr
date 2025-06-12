import numpy as np
import csv
from LogisticRegression import LogisticRegression
import sys

def load_data_from_csv(filename: str):
    features = []
    labels = []

    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            label = row["Hogwarts House"]
            labels.append(label)

            # Extract only the numeric features (from Arithmancy onward)
            sample_features = []
            for key in row:
                if key in ["Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts",
                           "Divination", "Muggle Studies", "Ancient Runes", "History of Magic",
                           "Transfiguration", "Potions", "Care of Magical Creatures", "Charms", "Flying"]:
                    val = row[key]
                    try:
                        sample_features.append(float(val))
                    except ValueError:
                        sample_features.append(0.0)  # handle missing or non-numeric values
            features.append(sample_features)

    input = np.array(features, dtype=np.float32)  # shape: (num_samples, input_dim)
    return input, labels


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 mainTrain.py <dataset.csv>")
        sys.exit(1)

    inputPath = sys.argv[1]

    labels = ["Ravenclaw", "Slytherin", "Hufflepuff", "Gryffindor"]
    inputs, inputsLabel = load_data_from_csv(inputPath)

    inputMean = inputs.mean(axis=0)
    inputStd = inputs.std(axis=0) + 1e-8
    inputs = (inputs - inputMean) / inputStd

    model = LogisticRegression(inputDim=inputs.shape[1], labels=labels)
    model.train(inputs, inputsLabel, epochs=100, lr=0.1)

    # Save model parameters and normalization
    np.save("model_weights.npy", model.layer.weights)
    np.save("model_bias.npy", model.layer.biais)
    np.save("model_labels.npy", np.array(labels))
    np.save("X_mean.npy", inputMean)
    np.save("X_std.npy", inputStd)