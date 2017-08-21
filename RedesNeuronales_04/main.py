import random



from RedesNeuronales_03.Perceptron import Perceptron
from RedesNeuronales_03.SigmoidNeuron import SigmoidNeuron
from RedesNeuronales_04.NeuralNetwork import NeuralNetwork


def main ():
    inputs = [1,2]
    neural_network = NeuralNetwork(inputs)
    neural_network.setRandomLayers(10, 3, 20)
    neural_network.addLastLayer()
    output = neural_network.getOutput()[0]
    print(output)











main()

