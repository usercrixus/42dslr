import sys
import numpy as np
from LogisticRegression import LogisticRegression
from params import selectedCathegory
from loader import getInput

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
    input = LogisticRegression.normalizeInfer(
        getInput(inputPath, selectedCathegory), inputMean, inputStd
    )

    # Build dummy model and inject weights
    model = LogisticRegression(inputDim=input.shape[1], labels=labels)
    model.loadModel(weights, biais)

    predictions = model.predict(input)
    print("Index,Hogwarts House")
    for i, pred in enumerate(predictions):
        print(f"{i},{pred}")
