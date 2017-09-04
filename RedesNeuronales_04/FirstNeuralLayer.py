from RedesNeuronales_04.AbstractNeuralLayer import AbstractNeuralLayer



class FirstNeuralLayer(AbstractNeuralLayer):

    def backPropagation(self,expected_output):
        self.calculateDelta(expected_output)
        self.forwardPropagation()

    def getOutputs(self,inputs):
        outputs = []
        for neuron in self.neuron_array:
            #print(inputs)
            neuron.setInputsList(inputs)
            neuron.output = neuron.getOutput()
            outputs.append(neuron.output)
        return self.next_layer.getOutputs(outputs)