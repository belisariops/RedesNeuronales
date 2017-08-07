import random

from RedesNeuronales_02.Perceptron import Perceptron


def main ():

    perceptron = Perceptron(0.01)
    perceptron.setBias(1)
    perceptron.setWeights(2,3)

    for i in range(100):
        x = random.randint(-100,100)
        y = random.randint(-100,100)
        real_output = False
        if 2*x +3 <= y:
            real_output = True
        perceptron.train(x,y,real_output)



main()

