from RedesNeuronales_01.NandPerceptron import NandPerceptron


class SummingValue:
    def __init__(self,sum,carry_bit):
        self.sum = sum
        self.carry_bit = carry_bit

class SummingNumberGate:
    def __init__(self,x1,x2):
        self.first_output = NandPerceptron(x1,x2).getOutput()
        self.second_output = NandPerceptron(x1,self.first_output).getOutput()
        self.third_output = NandPerceptron(self.first_output,x2).getOutput()
        self.carry_bit = NandPerceptron(self.first_output,self.first_output).getOutput()
        self.sum = NandPerceptron(self.second_output,self.third_output).getOutput()

    def getSummingValue(self):
        return SummingValue(self.sum, self.carry_bit)
