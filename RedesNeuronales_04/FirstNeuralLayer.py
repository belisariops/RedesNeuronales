from RedesNeuronales_04.AbstractNeuralLayer import AbstractNeuralLayer



class FirstNeuralLayer(AbstractNeuralLayer):

    def backPropagation(self,output,expected_output):
        self.forwardPropagation()
