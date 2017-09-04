from RedesNeuronales_04.AbstractNeuralLayer import AbstractNeuralLayer


class InnerNeuralLayer(AbstractNeuralLayer):
    def backPropagation(self,expected_output):
        self.calculateDelta(expected_output)
        self.previous_layer.backPropagation(expected_output)

    def getOutputs(self,inputs):
        outputs = []
        for neuron in self.neuron_array:
            neuron.setInputsList(inputs)
            neuron.output = neuron.getOutput()
            outputs.append(neuron.output)
        return self.next_layer.getOutputs(outputs)