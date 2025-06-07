"Layer take as many neuron as class to predict"

from typing import List
from srcs.model.Neuron import Neuron


class Layer:
    def __init__(self, neurons:List[Neuron]):
        ""