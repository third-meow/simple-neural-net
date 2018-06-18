from simple_nn import Network

training_input = [
        [0,0,1,1],
        [1,1,0,0],
        [0,1,0,1],
        [1,0,1,0],
        [0,0,0,1],
        [0,0,1,0],
        [0,1,0,0],
        [1,0,0,0],
        [1,1,1,1],
        [0,0,0,0],
        [1,0,0,1],
        [0,1,1,0],
        ]

training_input_labels = [3,3,2,2,1,1,1,1,0,0,4,4]



#create neural network
my_net = Network('a-net-of-lines',[4,6,6,6,5])
my_net.train(training_input, training_input_labels, 100000, vbos=True)

print(my_net.save())
