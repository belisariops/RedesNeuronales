from RedesNeuronales_04.FirstNeuralLayer import FirstNeuralLayer
from RedesNeuronales_04.LastNeuralLayer import LastNeuralLayer
from RedesNeuronales_04.NeuralNetwork import NeuralNetwork
from RedesNeuronales_04.SigmoidNeuron import SigmoidNeuron


def main ():
    inputs = [1,1]
    neural_network = NeuralNetwork(inputs)
    neural_network.setRandomLayers(1, 1, 1)
    # neural_layer1 = FirstNeuralLayer()
    # neuron1 = SigmoidNeuron(0.5)
    # neuron2 = SigmoidNeuron(0.5)
    # neuron3 = SigmoidNeuron(0.5)
    # neuron1.weights = [1,3]
    # neuron2.weights = [2,5]
    # neuron3.weights = [4,4]
    #
    #
    # neural_layer1.neuron_array = [neuron1,neuron2]
    # neural_layer2 = LastNeuralLayer()
    # neural_layer2.neuron_array = [neuron3]
    # neural_layer1.setNextLayer(neural_layer2)
    # neural_layer2.setPreviousLayer(neural_layer1)
    # neural_network.first_layer = neural_layer1
    # neural_network.output_layer = neural_layer2
    neural_network.train(100000,lambda x,y: x+y>8)

    output = neural_network.feed([0,1])
    print(output)











main()

