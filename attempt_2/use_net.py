from simple_nn import Network
import numpy
import matplotlib.pyplot as plt
train_input = numpy.array([
        [0,0,1,1],
        [1,1,0,0],
        [0,0,1,0],
        [0,0,0,1],
        [0,1,1,0],
        [1,0,0,1],
        [1,0,1,1],
        [1,1,0,1]])

train_labels = numpy.array([
    [3],
    [0],
    [3],
    [3],
    [0],
    [0],
    [3],
    [0]])



#create neural network
net = Network(0.1)
for i in range(7000):
    net.train(train_input, train_labels, vbos = True) 

foo = numpy.array([0,0,1,0])
print('0010 = {}'.format(net.predict(foo, True).T))
foo = numpy.array([1,0,0,0])
print('1000 = {}'.format(net.predict(foo, True).T))
