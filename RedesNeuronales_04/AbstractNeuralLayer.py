from RedesNeuronales_04.SigmoidNeuron import SigmoidNeuron
from abc import ABC, abstractmethod


class AbstractNeuralLayer(ABC):
    def __init__(self, neuron_array=None):
        self.neuron_array = neuron_array
        if neuron_array is None:
            self.neuron_array = []
        self.next_layer = None
        self.previous_layer = None

    def buildRandomLayer(self, number_of_neurons):
        neuron = None

        for i in range(number_of_neurons):
            neuron = SigmoidNeuron()
            neuron.setRandomParameters()
            self.neuron_array.append(neuron)

    

    def forwardPropagation(self):
        for neuron in self.neuron_array:
            neuron.updateWeights()
            neuron.updateBias()
        self.next_layer.forwardPropagation()


    def transferDerivative(self,output):
        return output*(1.0 - output)



    def setPreviousLayer(self, previous_layer):
        self.previous_layer = previous_layer

    def setNextLayer(self, next_layer):
        self.next_layer = next_layer

    def getNumberofNeurons(self):
        return len(self.neuron_array)

    @abstractmethod
    def getOutputs(self,inputs):
        pass

    def setRandomWeights(self,number_of_weights,min_value,max_value):
        for neuron in self.neuron_array:
            neuron.setRandomWeights(number_of_weights,min_value,max_value)

    def calculateDelta(self,expected_output):
        for index in range(len(self.neuron_array)):
            error = 0
            for next_neuron in self.next_layer.neuron_array:
                error += next_neuron.weights[index]*next_neuron.delta
            self.neuron_array[index].delta = error * self.transferDerivative(self.neuron_array[index].output)



