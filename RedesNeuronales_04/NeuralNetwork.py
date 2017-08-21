import random

from RedesNeuronales_04.NeuralLayer import NeuralLayer


class NeuralNetwork:
    def __init__(self, inputs=None):
        self.inputs = inputs
        self.neural_layers = None

    def setRandomLayers(self, number_of_layers, min_neurons_per_layer, max_neurons_per_layer):
        neural_layer = None
        previous_layer = None
        first_layer = None
        for i in range(number_of_layers):
            neural_layer = NeuralLayer()
            neural_layer.buildRandomLayer(random.randint(min_neurons_per_layer, max_neurons_per_layer))
            if previous_layer is None:
                inputs_next_layer = neural_layer.getNumberofNeurons()
                first_layer = neural_layer
                number_weights = len(self.inputs)
            else:
                previous_layer.setNextLayer(neural_layer)
                number_weights = previous_layer.getNumberofNeurons()
            neural_layer.setRandomWeights(number_weights, 1, 10)
            neural_layer.setPreviousLayer(previous_layer)
            previous_layer = neural_layer
        self.neural_layers = first_layer

    def feed(self, inputs):
        self.inputs = inputs

    def addLayer(self, neural_layer):
        self.neural_layers.append(neural_layer)

    def addRandomLayer(self, number_of_neurons):
        self.neural_layers.append(NeuralLayer().buildRandomLayer(number_of_neurons))

    def getOutput(self):
        return self.neural_layers.getOutputs(self.inputs)

    def addLastLayer(self):
        layer = self.neural_layers
        while layer is not None:
            current_layer = layer
            layer = layer.next_layer
        last_layer = NeuralLayer()
        last_layer.buildRandomLayer(1)
        last_layer.setRandomWeights(current_layer.getNumberofNeurons(), 1, 10)
        current_layer.setNextLayer(last_layer)
        last_layer.setPreviousLayer(current_layer)
