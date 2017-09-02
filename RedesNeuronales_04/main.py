from RedesNeuronales_04.FirstNeuralLayer import FirstNeuralLayer
from RedesNeuronales_04.LastNeuralLayer import LastNeuralLayer
from RedesNeuronales_04.NeuralNetwork import NeuralNetwork
from RedesNeuronales_04.SigmoidNeuron import SigmoidNeuron


def main ():
    inputs = [1,2]
    neural_network = NeuralNetwork(inputs)
   # neural_network.setRandomLayers(5, 1, 5)
    first_layer = FirstNeuralLayer()
    bias = 5
    neuron_0 = SigmoidNeuron()
    neuron_0.weights = [1,1.1]
    neuron_0.setC(0.2)
    neuron_0.setBias(bias)
    neuron_1 = SigmoidNeuron()
    neuron_1.weights = [-1, 2]
    neuron_1.setC(0.4)
    neuron_0.setBias(bias)
    neuron_2 = SigmoidNeuron()
    neuron_2.weights = [1, 2.3]
    neuron_2.setC(0.8)
    neuron_0.setBias(bias)
    first_layer.neuron_array = [neuron_0,neuron_1]
    last_layer = LastNeuralLayer()
    last_layer.neuron_array = [neuron_2]
    first_layer.setNextLayer(last_layer)
    last_layer.setPreviousLayer(first_layer)
    neural_network.first_layer = first_layer
    neural_network.output_layer = last_layer
    myfunc = lambda x,y: 1 if (x and y) else 0

    output = neural_network.feed([1,0])
    print(output)
    neural_network.train(1000,lambda x,y: 1 if (x and y) else 0)
    print(myfunc(0,0))

    output = neural_network.feed([1,0])
    print(output)











main()

