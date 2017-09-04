import random

import numpy

from RedesNeuronales_04.AbstractNeuron import AbstractNeuron


class SigmoidNeuron(AbstractNeuron):
    def __init__(self):
        super(SigmoidNeuron, self).__init__()
        self.activation_function = lambda z: (1.0 / (1.0 + numpy.exp(-1.0*z)))
        self.C = 0.05
        self.output = 0
        self.delta = 0

    def getOutput(self):
        z = 0
        for i in range(len(self.weights)):
            z = z + self.weights[i] * self.inputs[i]
        z = z + self.bias
        self.output = self.activation_function(z)
        return self.output

    def setRandomParameters(self):
        self.setBias(random.randint(1, 5))


