class Perceptron:
    def __init__(self,c):
        self.C  = c

    def setWeights(self,w1,w2):
        self.w1 = w1
        self.w2 = w2

    def setInputs(self,x1,x2):
        self.x1 = x1
        self.x2 = x2

    def setBias(self,bias):
        self.bias = bias

    def getOutput(self):
        return (self.x1*self.w1 + self.x2*self.w2 + self.bias > 0)

    def decreaseWeight(self):
        self.w1 = self.w1 - self.C*self.x1
        self.w2 = self.w2 - self.C*self.x2

    def increaseWeight(self):
        self.w1 = self.w1 + self.C * self.x1
        self.w2 = self.w2 + self.C * self.x2

    def train(self,x,y,output):
        self.setInputs(x,y)
        if not output and self.getOutput():
            self.decreaseWeight()

        if output and not self.getOutput():
            self.increaseWeight()



