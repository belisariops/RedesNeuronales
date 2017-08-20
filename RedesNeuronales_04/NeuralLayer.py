from RedesNeuronales_04.SigmoidNeuron import SigmoidNeuron


class NeuralLayer:
    def __init__(self,neuron_array = None):
        self.neuron_array = neuron_array
        if neuron_array is None:
            self.neuron_array = []
        self.next_layer = None
        self.previous_layer = None

    def buildRandomLayer(self,number_of_neurons):
        neuron = None
        for i in range(number_of_neurons):
            neuron = SigmoidNeuron(0.5)

    def setPreviousLayer(self,previous_layer):
        self.previous_layer = previous_layer

    def setNextLayer(self,next_layer):
        self.next_layer = next_layer

