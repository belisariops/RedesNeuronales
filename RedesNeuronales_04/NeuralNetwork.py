import random

from RedesNeuronales_04.InnerNeuralLayer import InnerNeuralLayer
from RedesNeuronales_04.LastNeuralLayer import LastNeuralLayer
from RedesNeuronales_04.FirstNeuralLayer import FirstNeuralLayer


class NeuralNetwork:
    def __init__(self, numberOfInputs):
        self.first_layer = None
        self.output_layer = None
        self.min_neurons_per_layer = 0
        self.max_neurons_per_layer = 0
        self.numberOfInputs = numberOfInputs

    def createLayer(self, neural_layer, previous_layer):
        neural_layer.buildRandomLayer(random.randint(self.min_neurons_per_layer, self.max_neurons_per_layer))
        if previous_layer is None:
            first_layer = neural_layer
            number_weights = self.numberOfInputs
        else:
            previous_layer.setNextLayer(neural_layer)
            number_weights = previous_layer.getNumberofNeurons()
        neural_layer.setRandomWeights(number_weights, -1, 3)
        neural_layer.setPreviousLayer(previous_layer)
        previous_layer = neural_layer
        return neural_layer

    def setRandomLayers(self, number_of_layers, min_neurons_per_layer, max_neurons_per_layer,number_of_outputs):
        first_layer = FirstNeuralLayer()
        neural_layer = first_layer
        self.min_neurons_per_layer = min_neurons_per_layer
        self.max_neurons_per_layer = max_neurons_per_layer
        previous_layer = self.createLayer(first_layer, None)
        for i in range(number_of_layers - 1):
            neural_layer = InnerNeuralLayer()
            previous_layer = self.createLayer(neural_layer, previous_layer)

        neural_layer = LastNeuralLayer()
        # neural_layer = self.createLayer(neural_layer,previous_layer)
        neural_layer.buildRandomLayer(number_of_outputs)
        neural_layer.setPreviousLayer(previous_layer)
        previous_layer.setNextLayer(neural_layer)
        neural_layer.setRandomWeights(len(neural_layer.previous_layer.neuron_array), -1, 3)

        self.output_layer = neural_layer

        self.first_layer = first_layer

    def setInputs(self, inputs):
        self.inputs = inputs

    def addLayer(self, neural_layer):
        self.first_layer.append(neural_layer)

    def addRandomLayer(self, number_of_neurons):
        self.first_layer.append(InnerNeuralLayer().buildRandomLayer(number_of_neurons))

    def feed(self, inputs):
        return self.first_layer.getOutputs(inputs)

    # def getOutput(self,inputs):

    def addLastLayer(self):
        layer = self.first_layer
        while layer is not None:
            current_layer = layer
            layer = layer.next_layer
        last_layer = LastNeuralLayer()
        last_layer.buildRandomLayer(1)
        last_layer.setRandomWeights(current_layer.getNumberofNeurons(), -3, 3)
        current_layer.setNextLayer(last_layer)
        last_layer.setPreviousLayer(current_layer)
        self.output_layer = last_layer

    def forwardPropagation(self):
        self.first_layer.forwardPropagation()

    def train(self, numberOfEpochs, data):
        for i in range(numberOfEpochs):
            for set in data:
                input_data = set[0:self.numberOfInputs]
                print(input_data)
                output_last_layer = self.feed(input_data)
                expected_output = set[-1:][0]
                self.output_layer.backPropagation(expected_output)
                self.forwardPropagation()

            # error = expected_output - output_last_layer
            # delta = error * (output_last_layer * (1.0 - output_last_layer))
