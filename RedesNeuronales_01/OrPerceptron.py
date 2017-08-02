from RedesNeuronales_01.Perceptron import Perceptron


class OrPerceptron:
    def __init__(self,x1,x2):
        self.perceptron = Perceptron()
        self.perceptron.setInputs(x1, x2)
        self.perceptron.setWeights(1, 1)
        self.perceptron.setBias(0)

    def getOutput(self):
        return self.perceptron.getOutput()
