import numpy as np
import csv
import sys
from LogisticRegression import LogisticRegression

def loadDataFromCsv(filename: str):
    features = []
    labels = []

    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            label = row["Hogwarts House"]
            labels.append(label)

            sample_features = []
            for key in row:
                if key in ["Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts",
                           "Divination", "Muggle Studies", "Ancient Runes", "History of Magic",
                           "Transfiguration", "Potions", "Care of Magical Creatures", "Charms", "Flying"]:
                    val = row[key]
                    try:
                        sample_features.append(float(val))
                    except ValueError:
                        sample_features.append(0.0)
            features.append(sample_features)

    inputs = np.array(features, dtype=np.float32)
    return inputs, labels


def trainValSplit(inputs: np.ndarray, labels: list[str], train_ratio=0.9):
    indices = np.arange(len(inputs))
    np.random.shuffle(indices)

    split = int(len(inputs) * train_ratio)
    train_idx = indices[:split]
    val_idx = indices[split:]

    X_train = inputs[train_idx]
    y_train = [labels[i] for i in train_idx]

    X_val = inputs[val_idx]
    y_val = [labels[i] for i in val_idx]

    return X_train, y_train, X_val, y_val


def computeAccuracy(model: LogisticRegression, X_val: np.ndarray, y_val: list[str]) -> float:
    predictions = model.predict(X_val)
    correct = sum(pred == true for pred, true in zip(predictions, y_val))
    return correct / len(y_val)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 mainTrain.py <dataset.csv>")
        sys.exit(1)

    inputPath = sys.argv[1]

    labels = ["Ravenclaw", "Slytherin", "Hufflepuff", "Gryffindor"]
    inputs, inputsLabel = loadDataFromCsv(inputPath)

    # Normalize
    inputMean = inputs.mean(axis=0)
    inputStd = inputs.std(axis=0) + 1e-8
    inputs = (inputs - inputMean) / inputStd

    # Split
    inputTrain, labelsTrain, inputVal, labelsVal = trainValSplit(inputs, inputsLabel, train_ratio=0.9)

    # Train
    model = LogisticRegression(inputDim=inputs.shape[1], labels=labels)
    model.train(inputTrain, labelsTrain, inputVal, labelsVal, epochs=200, lr=0.01, batch_size=32)

    # Save model
    np.save("model_weights.npy", model.layer.weights)
    np.save("model_bias.npy", model.layer.biais)
    np.save("model_labels.npy", np.array(labels))
    np.save("X_mean.npy", inputMean)
    np.save("X_std.npy", inputStd)

