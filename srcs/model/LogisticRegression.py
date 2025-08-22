import numpy as np
from Layer import Layer
from Regression import Regression


class LogisticRegression(Regression):
    def __init__(self, inputDim: int, labels: list[str]):
        self.labels = labels
        self.layer = Layer(inputDim, len(labels), labels)

    def computeBatch(self, batchIndices, inputTrain, labelTrain):
        gradW = np.zeros_like(self.layer.weights)
        gradB = np.zeros_like(self.layer.biais)
        batchLoss = 0
        for i in batchIndices:
            input = inputTrain[i]
            trueLabelVector = np.zeros(len(self.labels))
            trueLabelVector[self.layer.labelToIndex[labelTrain[i]]] = 1

            probabilities = self.layer.infer(input)
            batchLoss += -np.sum(trueLabelVector * np.log(probabilities + 1e-8))

            deltas = probabilities - trueLabelVector
            gradW += np.outer(deltas, input)
            gradB += deltas
        return gradW, gradB, batchLoss

    def train(
        self,
        inputTrain: np.ndarray,
        labelTrain: list[str],
        inputVal: np.ndarray,
        labelVal: list[str],
        epochs: int,
        lr: float,
        batch_size: int = 32,
    ):
        numSamples = inputTrain.shape[0]
        for epoch in range(epochs):
            indices = np.arange(numSamples)
            np.random.shuffle(indices)
            totalLoss = 0
            for start in range(0, numSamples, batch_size):
                end = min(start + batch_size, numSamples)
                batchIndices = indices[start:end]
                gradW, gradB, batchLoss = self.computeBatch(
                    batchIndices, inputTrain, labelTrain
                )
                totalLoss += batchLoss
                self.layer.weights -= lr * (gradW / len(batchIndices))
                self.layer.biais -= lr * (gradB / len(batchIndices))
            avgLoss = totalLoss / numSamples
            valAcc = self.computeAccuracy(inputVal, labelVal)
            print(
                f"Epoch {epoch + 1} | Loss: {avgLoss:.4f} | Val Accuracy: {valAcc * 100:.2f}%"
            )

    def computeAccuracy(self, X: np.ndarray, y: list[str]) -> float:
        predictions = self.predict(X)
        correct = sum(pred == true for pred, true in zip(predictions, y))
        return correct / len(y)

    def predict(self, input: np.ndarray) -> list[str]:
        predictions = []
        for i in range(input.shape[0]):
            row = input[i]
            rowPrediction = self.layer.infer(row)
            classIndex = np.argmax(rowPrediction)
            predictions.append(self.labels[classIndex])
        return predictions

    def loadModel(self, weights, biais):
        self.layer.weights = weights
        self.layer.biais = biais
