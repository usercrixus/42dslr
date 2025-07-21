import numpy as np
from Layer import Layer

class LogisticRegression:
    def __init__(self, inputDim: int, labels: list[str]):
        self.labels = labels
        self.layer = Layer(inputDim, len(labels), labels)

    def train(self, inputTrain: np.ndarray, labelTrain: list[str], inputVal: np.ndarray, labelVal: list[str], epochs: int, lr: float, batch_size: int = 32):
        numSamples = inputTrain.shape[0] # number of input in the train set

        for epoch in range(epochs):
            indices = np.arange(numSamples)
            np.random.shuffle(indices) # random the set for of input for better training

            totalLoss = 0
            for start in range(0, numSamples, batch_size):
                end = min(start + batch_size, numSamples)
                batchIndices = indices[start:end]

                gradW = np.zeros_like(self.layer.weights)
                gradB = np.zeros_like(self.layer.biais)

                for i in batchIndices:
                    input = inputTrain[i]
                    trueLabel = labelTrain[i]

                    probabilities = self.layer.infer(input)
                    trueLabelVector = np.zeros(len(self.labels))
                    trueLabelVector[self.layer.labelToIndex[trueLabel]] = 1 # we create a vector where all label are 0 exepect the true one (example [0, 0, 1, 0])

                    loss = -np.sum(trueLabelVector * np.log(probabilities + 1e-8))
                    totalLoss += loss

                    deltas = probabilities - trueLabelVector
                    gradW += np.outer(deltas, input) # charge buffer
                    gradB += deltas # charge buffer

                self.layer.weights -= lr * (gradW / len(batchIndices)) # update
                self.layer.biais -= lr * (gradB / len(batchIndices)) # update

            avgLoss = totalLoss / numSamples
            val_acc = self.compute_accuracy(inputVal, labelVal)
            print(f"Epoch {epoch + 1} | Loss: {avgLoss:.4f} | Val Accuracy: {val_acc * 100:.2f}%")


    def compute_accuracy(self, X: np.ndarray, y: list[str]) -> float:
        predictions = self.predict(X)
        correct = sum(pred == true for pred, true in zip(predictions, y))
        return correct / len(y)


    def predict(self, input: np.ndarray) -> list[str]:
        predictions = []
        for i in range(input.shape[0]):
            row = input[i]  # ← row i
            rowPrediction = self.layer.infer(row)
            classIndex = np.argmax(rowPrediction)
            predictions.append(self.labels[classIndex])
        return predictions
