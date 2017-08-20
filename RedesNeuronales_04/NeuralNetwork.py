from RedesNeuronales_04.NeuralLayer import NeuralLayer


class NeuralNetwork:
    def __init__(self,inputs = None):
        self.inputs = inputs
        self.neural_layers = []

    def setInput(self,inputs):
        self.inputs = inputs

    def addLayer(self,neural_layer):
        self.neural_layers.append(neural_layer)

    def addRandomLayer(self,number_of_neurons):
        self.neural_layers.append(NeuralLayer().buildRandomLayer(number_of_neurons))



