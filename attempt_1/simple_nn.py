import numpy as np
import random
import pickle

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
        #list of activations, biases, costs and weights
        self.acts = []
        self.biases = []
        self.costs = []
        self.weights = []
        
        #set inital activations, biases and weights
        for n in range(neuron_n):
            #all acts, costs and biases start at 0
            self.acts.append(0)
            self.costs.append(0)
            self.biases.append(0)

            #append sub-list to weights
            self.weights.append([])
            #set weights in sub-list randomly
            for w in range(prv_neuron_n):
                self.weights[n].append(random.random())
    

    #allow network to set values in layer
    def set(self, input_values):
        for a in range(len(self.acts)):
            self.acts[a] = sigmoid(input_values[a])


    #run layer on an input layer
    def run(self, input_layer):
        #for every neuron in layer
        for n in range(len(self.acts)):
            #reset activation
            self.acts[n] = 0
            
            #for every neuron in input layer
            for i in range(len(input_layer.acts)):
                #add input neuron's activation (multiplyed by it's repective weight) 
                self.acts[n] += input_layer.acts[i] * self.weights[n][i]
            
            #add repective bias
            self.acts[n] += self.biases[n]
            
            #apply sigmoid
            self.acts[n] = sigmoid(self.acts[n])


class Network():
    def __init__(self, id_no, config, trainrate=0.005):
        #training rate (how fast the network 'learns')
        self.train_rate = trainrate
        
        #network's id
        self.id = id_no

        #set seed
        random.seed(1357)

        #list of Layers
        self.layers = []

        #create first layer (the stimuli layer)
        self.layers.append(Layer(config[0], 0))  
        
        #create other layers
        for l in range(1, len(config)):
            self.layers.append(Layer(config[l], len(self.layers[l - 1].acts)))  


    #training network on training data and labels
    def train(self, training_data, training_labels, repeats, vbos=False):
        for o in range(repeats):
            #calculate costs accross network
            self.calc_costs(training_data[o % len(training_data)], training_labels[o
                % len(training_labels)])
            
            #apply costs to weights
            for l in range(1, len(self.layers)):
                for c in range(len(self.layers[l].costs)):
                    for w in range(len(self.layers[l].weights[c])):
                        self.layers[l].weights[c][w] += (self.layers[l].costs[c] 
                            * self.layers[l-1].acts[w] * self.train_rate)

    #calc costs
    def calc_costs(self, training_set, label):
        #get actual_output by runing network
        actual_output = self.run(training_set, raw=True)

        #create a perfect last layer result from training label
        correct_output = []
        for n in range(len(self.layers[-1].acts)):
            if n == label:
                correct_output.append(1.0)
            else:
                correct_output.append(0.0)

        #find cost of last layer using correct_output
        for n in range(len(self.layers[-1].costs)):
            self.layers[-1].costs[n] = (correct_output[n] 
                - actual_output[n])

        #find cost of other layers
        for l in reversed(range(len(self.layers)-1)):
            for n in range(len(self.layers[l].costs)):
                cost_count = 0
                for c in range(len(self.layers[l+1].costs)):
                    cost_count += (self.layers[l+1].costs[c]
                    * self.layers[l+1].weights[c][n])
                self.layers[l].costs[n] = cost_count
            


    #run nework based on input stimuli
    def run(self, input_stimuli, raw=False):
        #set stimuli Layer to input stimuli
        self.layers[0].set(input_stimuli)
        
        for l in range(1,len(self.layers)):
            #run layer on the previous layer
            self.layers[l].run(self.layers[l-1])

        if raw == False:
            #find largest activation in output layer
            max_act = max(self.layers[-1].acts)
        
            #return index of largest activation in output layer
            return (self.layers[-1].acts.index(max_act))
        else:
            #return the acts from last layer ('raw' output)
            return self.layers[-1].acts


    #save nerual network(as trained) to file
    def save(self):
        #create file name from self.id
        save_str = 'saved_models/'+str(self.id)+'.p'
        #create dict with data to be saved
        save_data = {
                    'layers' : self.layers,
                    'id' : self.id,
                    'train_rate' : self.train_rate
                    }
        #use pickle to save save_data
        with open(save_str, 'wb') as file:
            pickle.dump(save_data, file)
        #return id 
        return self.id

    def load(self, save_id):
        #create file name from input id
        save_str = 'saved_models/'+str(save_id)+'.p'
        #load save data
        with open(save_str, 'rb') as file:
            save_data = pickle.load(file)
        #set id, layers and train_rate to the save data
        self.id = save_data['id']
        self.layers = save_data['layers']
        self.train_rate = save_data['train_rate']



