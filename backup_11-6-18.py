import numpy as np

def sigmoid(x, derive=False):
    if derive:
        return x * (1 - x)
    else:
        return 1 / (1 + np.exp(-x))

def avrg(in_list):
    total = 0
    for i in in_list:
        total += i
    return total / len(in_list)

def to_column(in_list, col_num):
    return[row[col_num] for row in in_list]


class Layer():
    #neuron_n is how meny neurons in Layer
    def __init__(self, neuron_n, prv_neuron_n):
        #list of activations
        self.acts = []
        
        #list of biases
        self.biases = []

        #2D list of weights
        self.weights = []

        #list of costs
        self.costs = []
        
        #set inital activations, biases and weights
        for n in range(neuron_n):
            #append 0 to acts, costs and biases
            self.acts.append(0)
            self.costs.append(0)
            self.biases.append(0)


            #append list of weights to self.weights
            self.weights.append([])
            #set weights in sub list
            for w in range(prv_neuron_n):
                self.weights[n].append(1)
    

    #allow network to set values in layer
    def set(self, input_values):
        for a in range(len(self.acts)):
            self.acts[a] = sigmoid(input_values[a])


    #run layer on an input layer
    def run(self, input_layer):
        #for every neuron in layer
        for n in range(len(self.acts)):
            #reset act  to 0
            self.acts[n] = 0
            
            #for every neuron in input layer
            for i in range(len(input_layer.acts)):
                #add input neuron's activation (times it's repective weight) 
                #to local neuron
                self.acts[n] += input_layer.acts[i] * self.weights[n][i]
            
            #add repective bias
            self.acts[n] += self.biases[n]
            
            #apply sigmoid to itself
            self.acts[n] = sigmoid(self.acts[n])


    #print reperestion of layer onto screen
    def display(self):
        for n in self.acts:
            print('(',end='')
            print(n, end='')
            print(')   ',end='')




class Network():
    def __init__(self, config, trainrate=0.5):
        #training rate (how fast the network 'learns')
        self.train_rate = trainrate
        
        #list of Layers
        self.layers = []

        #create first layer (the stimuli layer)
        self.layers.append(Layer(config[0], 0))  
        
        #create other layers
        for l in range(1, len(config)):
            self.layers.append(Layer(config[l], len(self.layers[l - 1].acts)))  


    #training network on training data and labels
    def train(self, training_data, training_labels, repeats=1000, vbos=False):
        for r in range(repeats):
            #calculate the costs
            self.calc_costs(training_data[r % len(training_data)],
            training_labels[r % len(training_data)])

            #adjust weights for all layers but first (first has no previous layer)
            for l in range(1, len(self.layers)):
                for n in range(len(self.layers[l].weights)):
                    for w in range(len(self.layers[l].weights[n])):
                        #weight += relvevent cost * trainging rate
                        self.layers[l].weights[n][w] += (self.layers[l].costs[n] 
                        * self.train_rate)
            


    #calc network's cost
    def calc_costs(self, training_data, training_label):
        #set costs to 0
        for l in self.layers:
            for n in l.costs:
                n = 0.0

        #get actual output by running network in raw mode
        actual_output = self.run(training_data, raw=True)
        
        #create the correct output based on label
        correct_output = []
        for o in range(len(actual_output)):
            if o != training_label:
                correct_output.append(0.0)
            else:
                correct_output.append(1.0)
        
        #find last layer's costs against correct output
        for n in range(len(self.layers[-1].costs)):
            self.layers[-1].costs[n] = ((correct_output[n] - self.layers[-1].acts[n]) ** 2)            
        
        #calc costs of other layers
        for l, e in reversed(list(enumerate(self.layers))):
            if l < len(self.layers) - 1:
                for n in range(len(e.costs)):
                    self.layers[l].costs[n] = (
                    (avrg(to_column(self.layers[l+1].weights,n)) - self.layers[l].acts[n]) ** 2)            
    

    #run nework based on input stimuli
    def run(self, input_stimuli, raw=False):
        #set stimuli Layer to input stimuli
        self.layers[0].set(input_stimuli)
        
        #run layers[1] on the stimuli layer     #
        self.layers[1].run(self.layers[0])      # replace with for loop
                                                # in future
        #run layers[2] on layers[0]             #
        self.layers[2].run(self.layers[1])      
        print('.')

        if raw == False:
            #find largest activation in output layer
            max_act = max(self.layers[-1].acts)
        
            #return index of largest activation in output layer
            return (self.layers[-1].acts.index(max_act))
        else:
            return self.layers[-1].acts


    #display representation of network
    def display(self):
        for l in self.layers:
            l.display()
            print('\n')




training_input = [
        [0,0,0,1],
        [0,0,1,0],
        [0,1,0,0],
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1],
        [0,0,1,0],
        [0,1,0,0],
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1],
        [0,0,1,0],
        [0,1,0,0],
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1],
        [0,0,1,0],
        ]


training_input_labels = [0,1,2,3,2,1,0,1,2,3,2,1,0,1,2,3,2,1,0,1]




#create neural network
my_net = Network([4,4,4])
my_net.train(training_input, training_input_labels)
my_net.run([0,0,1,0], raw=True)
