from abc import ABC, abstractmethod

class AbstractNeuron(ABC):
    def setC(self, c):
        self.C = c

    def setThreshold(self,threshold):
        self.threshold  = threshold

    def setWeights(self, *weights):
        self.weights = list(weights)

    def setInputs(self, *inputs):
        self.inputs = list(inputs)

    def setBias(self, bias):
        self.bias = bias

    @abstractmethod
    def getOutput(self):
        pass

    def decreaseWeight(self):
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - self.C * self.inputs[i]

    def increaseWeight(self):
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] + self.C * self.inputs[i]

    def train(self, *inputs, output):
        self.setInputs(*inputs)
        if not output and self.getOutput():
            self.decreaseWeight()

        if output and not self.getOutput():
            self.increaseWeight()