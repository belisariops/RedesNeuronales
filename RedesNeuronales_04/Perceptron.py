from RedesNeuronales_03.AbstractNeuron import AbstractNeuron


class Perceptron(AbstractNeuron):
    """the Perceptron is an artificial neuron using the Heaviside step function as the activation function.
    Attributes:
        inputs (list(float)): values recived by the perceptron.
        C (float): Constant value to drecrease or increase a weight of the perceptron proportional to the input, during the training phase.
        weigths (list(float)): weights of the inputs recieved by the perceptron.
        bias (float).

    """
    def __init__(self,c):
        self.C  = c


    def getOutput(self):
        output = 0
        for i in range(len(self.inputs)):
            output = output + self.inputs[i]*self.weights[i]
        return output + self.bias > 0


