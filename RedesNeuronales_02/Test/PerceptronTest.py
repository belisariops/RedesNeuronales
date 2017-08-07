import unittest

from AndPerceptron import AndPerceptron
from NandPerceptron import NandPerceptron
from OrPerceptron import OrPerceptron

from RedesNeuronales_01.SummingNumberGate import SummingNumberGate


class PerceptronsTest(unittest.TestCase):
    def test_nand_perceptron(self):
        self.nand_perceptron = NandPerceptron(0, 0)
        self.assertTrue(self.nand_perceptron.getOutput())
        self.nand_perceptron = NandPerceptron(0, 1)
        self.assertTrue(self.nand_perceptron.getOutput())
        self.nand_perceptron = NandPerceptron(1, 0)
        self.assertTrue(self.nand_perceptron.getOutput())
        self.nand_perceptron = NandPerceptron(1, 1)
        self.assertFalse(self.nand_perceptron.getOutput())

    def test_and_perceptron(self):
        self.and_perceptron = AndPerceptron(0, 0)
        self.assertFalse(self.and_perceptron.getOutput())
        self.and_perceptron = AndPerceptron(0, 1)
        self.assertFalse(self.and_perceptron.getOutput())
        self.and_perceptron = AndPerceptron(1,0)
        self.assertFalse(self.and_perceptron.getOutput())
        self.and_perceptron = AndPerceptron(1,1)
        self.assertTrue(self.and_perceptron.getOutput())

    def test_or_perceptron(self):
        self.or_perceptron = OrPerceptron(0,0)
        self.assertFalse(self.or_perceptron.getOutput())
        self.or_perceptron = OrPerceptron(0,1)
        self.assertTrue(self.or_perceptron.getOutput())
        self.or_perceptron = OrPerceptron(1,0)
        self.assertTrue(self.or_perceptron.getOutput())
        self.or_perceptron = OrPerceptron(1,1)
        self.assertTrue(self.or_perceptron.getOutput())


class SummingGateTest(unittest.TestCase):
    def setUp(self):
        self.summing_gate = None

    def test_sum_zeros(self):
        self.summing_gate = SummingNumberGate(0,0)
        summing_gate_value = self.summing_gate.getSummingValue()
        self.assertEqual(summing_gate_value.sum,0)
        self.assertEqual(summing_gate_value.carry_bit,0)


    def test_sum_zero_and_one(self):
        self.summing_gate = SummingNumberGate(0, 1)
        summing_gate_value = self.summing_gate.getSummingValue()
        self.assertEqual(summing_gate_value.sum, 1)
        self.assertEqual(summing_gate_value.carry_bit, 0)

    def test_sum_one_and_zero(self):
        self.summing_gate = SummingNumberGate(1, 0)
        summing_gate_value = self.summing_gate.getSummingValue()
        self.assertEqual(summing_gate_value.sum, 1)
        self.assertEqual(summing_gate_value.carry_bit, 0)

    def test_sum_one_and_one(self):
        self.summing_gate = SummingNumberGate(1, 1)
        summing_gate_value = self.summing_gate.getSummingValue()
        self.assertEqual(summing_gate_value.sum, 0)
        self.assertEqual(summing_gate_value.carry_bit, 1)

