import numpy

from RedesNeuronales_03.AbstractNeuron import AbstractNeuron


class SigmoidNeuron(AbstractNeuron):
    def __init__(self,threshold):
        self.activation_function = lambda z: 1/(1+pow(numpy.exp,-z))
        self.setThreshold(threshold)

    def getOutput(self):
        z = 0
        for i in range(len(self.weights)):
            z = z + self.weights[i]*self.inputs[i]
        z = z + self.bias
        output = self.activation_function(z)
        return output > self.threshold
