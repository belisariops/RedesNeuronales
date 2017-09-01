from RedesNeuronales_04.AbstractNeuralLayer import AbstractNeuralLayer



class FirstNeuralLayer(AbstractNeuralLayer):

    def backPropagation(self,expected_output):
        self.forwardPropagation()

    def getOutputs(self,inputs):
        outputs = []
        for neuron in self.neuron_array:
            neuron.setInputsList(inputs)
            outputs.append(neuron.getOutput())
        return self.next_layer.getOutputs(outputs)