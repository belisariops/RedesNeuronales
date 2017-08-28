from RedesNeuronales_04.AbstractNeuralLayer import AbstractNeuralLayer


class LastNeuralLayer(AbstractNeuralLayer):
    def backPropagation(self, expected_output):
        neuron = self.neuron_array[0]
        error = expected_output - neuron.output
        neuron.delta = error * (neuron.output * (1.0 - neuron.output))
        self.previous_layer.backPropagation()
