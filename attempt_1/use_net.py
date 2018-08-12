from simple_nn import Network

training_input = [
        [0,0,1,1],
        [1,1,0,0],
        [0,0,1,0],
        [0,1,0,0],
        [0,1,1,1],
        [1,0,1,0],
        [0,1,0,1],
        [1,0,1,1],
        ]

training_input_labels = [1,0,1,0,1,0,0,1]



#create neural network
my_net = Network('werid_net',[4,6,6,2])
my_net.train(training_input, training_input_labels, 1000000, vbos=True)

print(my_net.save())
