class Perceptron:
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

