import numpy as np
import sys
from LogisticRegression import LogisticRegression
from params import selectedCathegory, labels
from loader import getInputAndLabel


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 mainTrain.py <dataset.csv>")
        sys.exit(1)
    inputPath = sys.argv[1]

    inputs, inputsLabel = getInputAndLabel(inputPath, selectedCathegory)
    inputMean, inputStd, inputs = LogisticRegression.normalizeTrain(inputs)
    inputTrain, labelsTrain, inputVal, labelsVal = LogisticRegression.trainValSplit(
        inputs, inputsLabel, train_ratio=0.9
    )
    model = LogisticRegression(inputDim=inputs.shape[1], labels=labels)
    model.train(
        inputTrain, labelsTrain, inputVal, labelsVal, epochs=200, lr=0.1, batch_size=64
    )

    # Save model
    np.save("model_weights.npy", model.layer.weights)
    np.save("model_bias.npy", model.layer.biais)
    np.save("model_labels.npy", np.array(labels))
    np.save("X_mean.npy", inputMean)
    np.save("X_std.npy", inputStd)
