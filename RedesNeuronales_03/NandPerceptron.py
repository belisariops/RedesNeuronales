from RedesNeuronales_03.Perceptron import Perceptron


class NandPerceptron:
    def __init__(self,x1,x2):
        self.perceptron = Perceptron()
        self.perceptron.setInputs(x1, x2)
        self.perceptron.setWeights(-2, -2)
        self.perceptron.setBias(3)

    def getOutput(self):
        return self.perceptron.getOutput()


