import random
import matplotlib.pylab as plt


from RedesNeuronales_02.Perceptron import Perceptron


def main ():

    perceptron = Perceptron(0.01)
    perceptron.setBias(1)
    perceptron.setWeights(2,3)
    pointsX = []
    pointsY = []
    colors = []

    for i in range(1000):
        x = random.randint(-100,100)
        y = random.randint(-100,100)
        real_output = False
        if 2*x +3 <= y:
            real_output = True
        perceptron.train(x,y,real_output)


    for j in range(100):
        x = random.randint(-100,100)
        y = random.randint(-100,100)
        pointsX.append(x)
        pointsY.append(y)
        perceptron.setInputs(x,y)
        if perceptron.getOutput():
            colors.append('blue')
            continue
        else:
            colors.append('red')


    plt.scatter(pointsX,pointsY,s=50,marker='o',c=colors)
    plt.plot
    plt.show()






main()

