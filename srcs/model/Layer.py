import numpy as np

class Layer:
    def __init__(self, inputDim: int, numClasses: int, labels:list[str]):
        self.weights = np.random.randn(numClasses, inputDim)
        self.biais = np.zeros(numClasses)
        self.labelToIndex = {label: idx for idx, label in enumerate(labels)}

    def softmax(self, predictions: np.ndarray) -> np.ndarray:
        predictions = predictions - np.max(predictions)
        exp_preds = np.exp(predictions)
        return exp_preds / np.sum(exp_preds)

    def infer(self, input: np.ndarray) -> np.ndarray:
        predictions = np.dot(self.weights, input) + self.biais
        return self.softmax(predictions)
