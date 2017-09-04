import random

from RedesNeuronales_04.FirstNeuralLayer import FirstNeuralLayer
from RedesNeuronales_04.LastNeuralLayer import LastNeuralLayer
from RedesNeuronales_04.NeuralNetwork import NeuralNetwork
from RedesNeuronales_04.SigmoidNeuron import SigmoidNeuron


def main():
    data = [[0, 0, [1,0]], [0, 1,[0,1]], [1, 0,[0,1]], [1, 1, [1,0]]]
    neural_network = NeuralNetwork(2)
    #neural_network.setRandomLayers(2, 1, 5, 2)
    first_layer = FirstNeuralLayer()
    learningRate = 0.5
    bias = random.uniform(1, 8)
    neuron_0 = SigmoidNeuron()
    neuron_0.weights = [1, 1.1]
    neuron_0.setC(0.01)  # 0.2)
    neuron_0.setBias(bias)
    neuron_1 = SigmoidNeuron()
    neuron_1.weights = [-1, 2]
    neuron_1.setC(0.05)  # 4#)
    neuron_0.setBias(bias)
    neuron_2 = SigmoidNeuron()
    neuron_2.weights = [1, 2.3]
    neuron_2.setC(0.04)  # 8)
    neuron_3 = SigmoidNeuron()
    neuron_3.weights = [-1.2, 1.3,2.2]
    neuron_3.setC(0.4)  # 8)
    neuron_4 = SigmoidNeuron()
    neuron_4.weights = [1, 1.5,-1]
    neuron_4.setC(0.3)  # 8)
    first_layer.neuron_array = [neuron_0, neuron_1,neuron_2]
    last_layer = LastNeuralLayer()
    last_layer.neuron_array = [neuron_3,neuron_4]
    first_layer.setNextLayer(last_layer)
    last_layer.setPreviousLayer(first_layer)
    neural_network.first_layer = first_layer
    neural_network.output_layer = last_layer
    # myfunc = lambda x, y: 1 if (x != y) else 0
    # output = neural_network.feed([1, 1])
    #print(output)
    neural_network.train(100000, data)
    #print(myfunc(0, 0))

    output = neural_network.feed([1, 1])
    print(output)


main()
