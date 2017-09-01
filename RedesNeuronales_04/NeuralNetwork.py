import random

from RedesNeuronales_04.InnerNeuralLayer import InnerNeuralLayer
from RedesNeuronales_04.LastNeuralLayer import LastNeuralLayer
from RedesNeuronales_04.FirstNeuralLayer import FirstNeuralLayer



class NeuralNetwork:
    def __init__(self, inputs=None):
        self.inputs = inputs
        self.first_layer = None
        self.output_layer = None
        self.min_neurons_per_layer = 0
        self.max_neurons_per_layer = 0

    def createLayer(self,neural_layer,previous_layer):
        neural_layer.buildRandomLayer(random.randint(self.min_neurons_per_layer, self.max_neurons_per_layer))
        if previous_layer is None:
            first_layer = neural_layer
            number_weights = len(self.inputs)
        else:
            previous_layer.setNextLayer(neural_layer)
            number_weights = previous_layer.getNumberofNeurons()
        neural_layer.setRandomWeights(number_weights, 0, 10)
        neural_layer.setPreviousLayer(previous_layer)
        previous_layer = neural_layer
        return neural_layer


    def setRandomLayers(self, number_of_layers, min_neurons_per_layer, max_neurons_per_layer):
        first_layer = FirstNeuralLayer()
        neural_layer = first_layer
        self.min_neurons_per_layer = min_neurons_per_layer
        self.max_neurons_per_layer = max_neurons_per_layer
        previous_layer = self.createLayer(first_layer,None)
        for i in range(number_of_layers -1):
            neural_layer = InnerNeuralLayer()
            previous_layer = self.createLayer(neural_layer,previous_layer)

        neural_layer = LastNeuralLayer()
        #neural_layer = self.createLayer(neural_layer,previous_layer)
        neural_layer.buildRandomLayer(1)
        neural_layer.setPreviousLayer(previous_layer)
        previous_layer.setNextLayer(neural_layer)
        neural_layer.setRandomWeights(len(neural_layer.previous_layer.neuron_array),1,10)

        self.output_layer = neural_layer

        self.first_layer = first_layer
    def setInputs(self, inputs):
        self.inputs = inputs

    def addLayer(self, neural_layer):
        self.first_layer.append(neural_layer)

    def addRandomLayer(self, number_of_neurons):
        self.first_layer.append(InnerNeuralLayer().buildRandomLayer(number_of_neurons))

    def feed(self,inputs):
        return self.first_layer.getOutputs(inputs)

    def addLastLayer(self):
        layer = self.first_layer
        while layer is not None:
            current_layer = layer
            layer = layer.next_layer
        last_layer = LastNeuralLayer()
        last_layer.buildRandomLayer(1)
        last_layer.setRandomWeights(current_layer.getNumberofNeurons(), 1, 10)
        current_layer.setNextLayer(last_layer)
        last_layer.setPreviousLayer(current_layer)
        self.output_layer = last_layer

    def train(self,train_iterations,expected_function):
        for i in range(train_iterations):
            expected_output = 0 #Calcular expectedoutput
            input = [random.randint(0,1), random.randint(0,1)] #Some input
            output_last_layer = self.feed(input)
            expected_output = expected_function(input[0],input[1])
            if (input[0]!= input[1]):
                expected_output = 1
            else:
                expected_output = 0
            self.output_layer.backPropagation(expected_output)

            #error = expected_output - output_last_layer
            #delta = error * (output_last_layer * (1.0 - output_last_layer))

