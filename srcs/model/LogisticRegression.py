import numpy as np
from Layer import Layer

class LogisticRegression:
    def __init__(self, inputDim: int, labels: list[str]):
        self.labels = labels
        self.layer = Layer(inputDim, len(labels), labels)

    def train(self, inputs: np.ndarray, labels: list[str], epochs: int, lr: float):

        numSamples = inputs.shape[0]

        for epoch in range(epochs):
            totalLoss = 0
            for i in range(numSamples):
                input = inputs[i]
                trueLabel = labels[i]

                probabilities = self.layer.infer(input)

                trueLabelVector = np.zeros(len(self.labels))
                trueLabelVector[self.layer.labelToIndex[trueLabel]] = 1

                loss = -np.sum(trueLabelVector * np.log(probabilities + 1e-8))
                totalLoss += loss

                self.layer.backprop(input, trueLabelVector, probabilities, lr)

            avgLoss = totalLoss / numSamples
            print(f"Epoch {epoch + 1} | Loss: {avgLoss:.4f}")


    def predict(self, input: np.ndarray) -> list[str]:
        predictions = []
        for i in range(input.shape[0]):
            row = input[i]  # ← row i
            rowPrediction = self.layer.infer(row)
            classIndex = np.argmax(rowPrediction)
            predictions.append(self.labels[classIndex])
        return predictions
