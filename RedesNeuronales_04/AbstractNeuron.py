import random
from abc import ABC, abstractmethod
import matplotlib.pylab as plt
import numpy as np

class AbstractNeuron(ABC):
    def __init__(self):
        self.delta = 0
        self.inputs = []
        self.bias = 1
        self.C = 0.01

    def setC(self, c):
        self.C = c

    def setThreshold(self,threshold):
        self.threshold  = threshold

    def setWeights(self, *weights):
        self.weights = list(weights)

    def setInputs(self, *inputs):
        self.inputs = list(inputs)

    def replaceInputs(self,inputs):
        self.inputs = inputs

    def setInputsList(self,inputs):
        self.inputs = inputs


    def setWeight(self,w):
        self.weights.append(w)

    def setInput(self,i):
        self.inputs.append(i)

    def setBias(self, bias):
        self.bias = bias

    def setRandomWeights(self,number_of_weights, min, max):
        self.weights = []
        for i in range(number_of_weights):
            self.weights.append(random.uniform(min, max))



    @abstractmethod
    def getOutput(self):
        pass

    def setWeightInPosition(self,i,value):
        self.weights[i] += value

    def updateWeights(self):
        for i in range (len(self.weights)):
            self.weights[i] += (self.C * self.delta * self.inputs[i])

    def updateBias(self):
        self.bias += (self.C * self.delta)

    def decreaseWeight(self):
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - self.C * self.inputs[i]

    def increaseWeight(self):
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] + self.C * self.inputs[i]

    def trainIteration(self,func,times):
        for i in range(times):
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            real_output = False
            if func(x,y):
                real_output = True
            self.tweakWeights(x, y, output=real_output)

    def tweakWeights(self, *inputs, output):
        self.setInputs(*inputs)
        if not output and self.getOutput():
            self.decreaseWeight()

        if output and not self.getOutput():
            self.increaseWeight()

    def train(self,iterations,func):
        self.errors = []
        self.iteration = []
        for i in range(iterations):
            self.setC(0.001)
            self.setThreshold(0.5)
            self.setBias(2)
            self.setWeights(4, 4)
            # Entrenar i veces, con la funcion para obetner el output verdadero func.
            self.trainIteration(func, i)

            misses = 0
            for j in range(100):
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                self.setInputs(x, y)
                real_output = False
                if func(x,y):
                    real_output = True
                if self.getOutput() != real_output:
                    misses = misses + 1
            error = misses / 100
            self.errors.append(error)
            self.iteration.append(i + 1)

    def run(self,iterations):
        self.pointsX = []
        self.pointsY = []
        self.colors = []
        for j in range(iterations):
            x = random.randint(0, 100)
            y = random.randint(0, 100)
            self.pointsX.append(x)
            self.pointsY.append(y)
            self.setInputs(x, y)
            if self.getOutput():
                self.colors.append('blue')
            else:
                self.colors.append('red')
    def plotResults(self,first,second):
        t = np.arange(0., 52., 1)
        plt.figure()
        plt.title(first,fontsize=20)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.plot(t, 2 * t + 3, c='black')
        plt.scatter(self.pointsX, self.pointsY, s=50, marker='o', c=self.colors)


        plt.figure()
        plt.title(second, fontsize=20)
        plt.xlabel('iteration')
        plt.ylabel('misses percentage')
        plt.plot(self.iteration, self.errors, 'k')
        return plt
