import numpy as np

class Neuron:
    def __init__(self, nx:int):
        """
        Initialize a neuron instance.

        Parameters:
        nx (int): Number of input features.

        Attributes:
        _w (np.ndarray): Weights vector for the neuron.
        _b (float): Bias for the neuron.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self._w = np.random.randn(1, nx)
        self._b = 0