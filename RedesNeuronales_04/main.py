from RedesNeuronales_04.NeuralNetwork import NeuralNetwork


def main ():
    inputs = [20,230]
    neural_network = NeuralNetwork(inputs)
    neural_network.setRandomLayers(3, 1, 3)

    neural_network.train(10000,lambda x,y: x+y>8)

    output = neural_network.feed([1,1])
    print(output)











main()

