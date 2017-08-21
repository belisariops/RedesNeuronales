from RedesNeuronales_04.SigmoidNeuron import SigmoidNeuron


class NeuralLayer:
    def __init__(self, neuron_array=None):
        self.neuron_array = neuron_array
        if neuron_array is None:
            self.neuron_array = []
        self.next_layer = None
        self.previous_layer = None

    def buildRandomLayer(self, number_of_neurons, threshold=None):
        neuron = None
        if threshold is None:
            threshold = 0.5

        for i in range(number_of_neurons):
            neuron = SigmoidNeuron(threshold)
            neuron.setRandomParameters()
            self.neuron_array.append(neuron)


    def setPreviousLayer(self, previous_layer):
        self.previous_layer = previous_layer

    def setNextLayer(self, next_layer):
        self.next_layer = next_layer

    def getNumberofNeurons(self):
        return len(self.neuron_array)

    def getOutputs(self,inputs):
        outputs = []
        for neuron in self.neuron_array:
            neuron.setInputsList(inputs)
            outputs.append(neuron.getOutput())
        if self.next_layer is None:
            return outputs
        return self.next_layer.getOutputs(outputs)



    def setRandomWeights(self,number_of_weights,min_value,max_value):
        for neuron in self.neuron_array:
            neuron.setRandomWeights(number_of_weights,min_value,max_value)
