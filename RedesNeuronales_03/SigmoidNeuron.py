import numpy


from RedesNeuronales_03.AbstractNeuron import AbstractNeuron


class SigmoidNeuron(AbstractNeuron):
    def __init__(self,threshold):
        
        self.setThreshold(threshold)
        self.activation_function = lambda z,threshold: (1/(1+numpy.exp(-z))) > self.threshold

    def getOutput(self):
        z = 0
        for i in range(len(self.weights)):
            z = z + self.weights[i]*self.inputs[i]
        z = z + self.bias
        output = self.activation_function(z,self.threshold)
        return output
