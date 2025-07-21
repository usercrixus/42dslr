import numpy as np

class Layer:
    def __init__(self, inputDim: int, numClasses: int, labels:list[str]):
        self.weights = np.random.randn(numClasses, inputDim) # init weight for each labels
        self.biais = np.zeros(numClasses) # init biais for each labels
        self.labelToIndex = {label: idx for idx, label in enumerate(labels)} # link each label with its id

    def softmax(self, predictions: np.ndarray) -> np.ndarray:
        predictions = predictions - np.max(predictions)
        exp_preds = np.exp(predictions)
        return exp_preds / np.sum(exp_preds)

    def infer(self, input: np.ndarray) -> np.ndarray:
        predictions = np.dot(self.weights, input) + self.biais # classic linear regression infer function
        return self.softmax(predictions) # return soft max
