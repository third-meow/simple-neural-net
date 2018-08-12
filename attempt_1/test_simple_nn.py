import unittest
from simple_nn import Network


training_input = [
              [0,0,0,1],
              [1,0,0,0],
              [0,1,0,0],
              [0,0,1,0],
              [0,0,0,1],
              [1,0,0,0],
              [0,0,1,0],
              [0,1,0,0],
              [1,0,0,0],
              [0,0,0,1],
              [0,1,0,0],
              [0,1,0,0],
              [0,0,0,1],
              [0,0,0,1],
              [0,1,0,0],
              [0,0,1,0],
              [0,0,0,1],
              [1,0,0,0],
              [0,1,0,0],
              [0,0,1,0],
              ]

training_input_labels = [1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0]







class TestNetwork(unittest.TestCase):
    def build (self):
        self.my_6net = Network('model-6',[4,2])
        self.my_6net.train(training_input, training_input_labels, 20000)

        self.my_10net = Network('model-10',[4,4,2])
        self.my_10net.train(training_input, training_input_labels, 20000)

        self.my_14net = Network('model-14',[4,4,4,2])
        self.my_14net.train(training_input, training_input_labels, 20000)


    def test_6_run(self):
        self.build()
        self.assertAlmostEqual(self.my_6net.run([0,0,0,1]), 1)
        self.assertAlmostEqual(self.my_6net.run([0,0,1,0]), 0)
        self.assertAlmostEqual(self.my_6net.run([0,1,0,0]), 0)
        self.assertAlmostEqual(self.my_6net.run([1,0,0,0]), 1)


    def test_10_run(self):
        self.build()
        self.assertAlmostEqual(self.my_10net.run([0,0,0,1]), 1)
        self.assertAlmostEqual(self.my_10net.run([0,0,1,0]), 0)
        self.assertAlmostEqual(self.my_10net.run([0,1,0,0]), 0)
        self.assertAlmostEqual(self.my_10net.run([1,0,0,0]), 1)


    def test_14_run(self):
        self.build()
        self.assertAlmostEqual(self.my_14net.run([0,0,0,1]), 1)
        self.assertAlmostEqual(self.my_14net.run([0,0,1,0]), 0)
        self.assertAlmostEqual(self.my_14net.run([0,1,0,0]), 0)
        self.assertAlmostEqual(self.my_14net.run([1,0,0,0]), 1)
