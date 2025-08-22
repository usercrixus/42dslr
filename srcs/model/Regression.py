import numpy as np


class Regression:

    def __init__(self):
        pass

    @staticmethod
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

    @staticmethod
    def normalizeTrain(inputs: np.array):
        inputMean = inputs.mean(axis=0)
        inputStd = inputs.std(axis=0) + 1e-8
        inputs = (inputs - inputMean) / inputStd
        return inputMean, inputStd, inputs

    @staticmethod
    def normalizeInfer(input: np.array, inputMean, inputStd):
        return (input - inputMean) / (inputStd + 1e-8)
