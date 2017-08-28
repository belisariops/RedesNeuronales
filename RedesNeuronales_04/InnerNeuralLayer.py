from RedesNeuronales_04.AbstractNeuralLayer import AbstractNeuralLayer


class InnerNeuralLayer(AbstractNeuralLayer):
    def backPropagation(self):
        for index in range(len(self.neuron_array)):
            error = 0
            for next_neuron in self.next_layer.neuron_array:
                error += next_neuron.weights[index]*next_neuron.delta
            self.neuron_array[index].delta = error * (self.neuron_array[index].output * (1.0 - self.neuron_array[index].output))
        self.previous_layer.propagate()
