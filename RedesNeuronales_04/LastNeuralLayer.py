from RedesNeuronales_04.AbstractNeuralLayer import AbstractNeuralLayer
from RedesNeuronales_04.SigmoidNeuron import SigmoidNeuron


class LastNeuralLayer(AbstractNeuralLayer):

    def backPropagation(self, expected_output):
        index = 0
        for neuron in self.neuron_array:
            error = expected_output[index] - neuron.output
            neuron.delta = error * self.transferDerivative(neuron.output)
            index +=1
        self.previous_layer.backPropagation(expected_output)

    def getOutputs(self, inputs):
        outputs = []
        for neuron in self.neuron_array:
            neuron.setInputsList(inputs)
            neuron.output = neuron.getOutput()
            outputs.append(neuron.output)

        return outputs

    def forwardPropagation(self):
        for neuron in self.neuron_array:
            neuron.updateWeights()
            neuron.updateBias()
