from RedesNeuronales_04.AbstractNeuralLayer import AbstractNeuralLayer
from RedesNeuronales_04.SigmoidNeuron import SigmoidNeuron


class LastNeuralLayer(AbstractNeuralLayer):

    def backPropagation(self, expected_output):
        neuron = self.neuron_array[0]
        error = expected_output - neuron.output
        neuron.delta = error * (neuron.output * (1.0 - neuron.output))
        self.previous_layer.backPropagation(expected_output)

    def getOutputs(self, inputs):
        outputs = []
        neuron = self.neuron_array[0]
        neuron.setInputsList(inputs)
        neuron.output = neuron.getOutput()
        return neuron.output

    def forwardPropagation(self):
        if len(self.neuron_array)>1:
            x =2
        for neuron in self.neuron_array:
            neuron.updateWeights()
            neuron.updateBias()
