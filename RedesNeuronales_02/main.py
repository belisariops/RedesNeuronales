import random
import matplotlib.pylab as plt
import numpy as np


from RedesNeuronales_02.Perceptron import Perceptron


def main ():

    perceptron = Perceptron(0.01)
    perceptron.setBias(1)
    perceptron.setWeights(2,3)
    pointsX = []
    pointsY = []
    colors = []
    errors = []
    iteration  = []
    count =1
    misses  = 0
    for i in range(10000):
        x = random.randint(-100,100)
        y = random.randint(-100,100)
        real_output = False
        if 2*x +3 <= y:
            real_output = True
        perceptron.train(x,y,output=real_output)

        if perceptron.getOutput() != real_output:
            misses = misses +1
        errors.append(misses / count)
        iteration.append(i)
        count = count +1


    for j in range(100):
        x = random.randint(0,100)
        y = random.randint(0,100)
        pointsX.append(x)
        pointsY.append(y)
        perceptron.setInputs(x,y)
        if perceptron.getOutput():
            colors.append('blue')
            continue
        else:
            colors.append('red')

    t = np.arange(0., 52., 1)
    plt.plot(t,2*t+3,c='black')
    plt.scatter(pointsX,pointsY,s=50,marker='o',c=colors)

    plt.title('Learning Perceptron',fontsize=20)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.figure()
    plt.scatter(iteration,errors)

    plt.show()






main()

