

class Perceptron:
    """the Perceptron is an artificial neuron using the Heaviside step function as the activation function.
    Attributes:
        inputs (list(float)): values recived by the perceptron.
        C (float): Constant value to drecrease or increase a weight of the perceptron proportional to the input, during the training phase.
        weigths (list(float)): weights of the inputs recieved by the perceptron.
        bias (float).

    """
    def __init__(self,c):
        self.C  = c

    def setC(self,c):
        self.C  = c

    def setWeights(self,*weights):
        self.weights = list(weights)


    def setInputs(self,*inputs):
        self.inputs = list(inputs)

    def setBias(self,bias):
        self.bias = bias


    def getOutput(self):
        output = 0
        for i in range(len(self.inputs)):
            output = output + self.inputs[i]*self.weights[i]
        return output + self.bias > 0

    def decreaseWeight(self):
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - self.C*self.inputs[i]

    def increaseWeight(self):
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] + self.C*self.inputs[i]

    def train(self,*inputs,output):
        self.setInputs(*inputs)
        if not output and self.getOutput():
            self.decreaseWeight()

        if output and not self.getOutput():
            self.increaseWeight()



