
import numpy as np

def sigmoid(x, derive=False):
    if derive:
        return x * (1 - x)
    else:
        return 1 / (1 + np.exp(-x))


def barr():
    return ('*' * 99)

class Network:
    #init nodes, weights etc
    def __init__(self, learnrate):
        #learning rate
        self.learnrate = learnrate

        #node layers            #nodes-per-layer and layer count is set for now
        self.input_layer = np.zeros([4])
        self.hidden_layer = np.zeros([4])
        self.output_layer = np.zeros([4])
        
        #weights
        self.weights_0 = np.random.rand(self.hidden_layer.size,
                self.input_layer.size) - 0.5
        
        self.weights_1 = np.random.rand(self.hidden_layer.size,
                self.input_layer.size) - 0.5
    


    #train net on training data
    def train(self, data, labels, vbos=False):
        #loop through all training data
        for i in range(data.shape[0]):
            #build np array represent the correct
            #raw output based on the label
            correct = [0,0,0,0]
            correct[labels[i][0]] = 1
            correct = np.array(correct, ndmin=2).T
            #get prediction from network
            prediction = self.predict(data[i], raw=True)

            if vbos and i % 1000 == 0:
                print(barr())
                print(barr())
                print('Training data {}'.format(data[i]))
                print('Correct arr {}'.format(correct))
                print('Prediction {}'.format(prediction))
        
            #get output layer error by subtracting prediction from correct
            output_layer_error = correct - prediction
            #get hidden layer error by 'running' output layer
            #error through second set of weights
            hidden_layer_error = np.dot(self.weights_1.T, output_layer_error)

            if vbos and i % 1000 == 0:
                print('Output error\n{}'.format(output_layer_error))
                print('Hidden error\n{}'.format(hidden_layer_error))
            
                print(barr())
                print('Output error\n{}'
                   .format(output_layer_error))
                print('Prediction\n{}'
                   .format(prediction))
                print('One minus prediction\n{}'
                   .format(1.0 - prediction))
                print('Hidden layer transposed {}'
                   .format(self.hidden_layer.T))
                print('Output error * prediction * 1-prediction\n{}'
                   .format(output_layer_error * prediction * (1.0 - prediction)))
                print('.. dot hidden outputs * lr\n{}'
                   .format(self.learnrate * np.dot(output_layer_error * 
                       prediction * (1.0 - prediction), self.hidden_layer.T)))
            
            #update weights
            self.weights_1 += (self.learnrate * 
                np.dot((output_layer_error * prediction * (1.0 - prediction)),
                    self.hidden_layer.T))

            self.weights_0 += (self.learnrate * 
                np.dot((hidden_layer_error * self.hidden_layer * 
                    (1.0 - self.hidden_layer)), self.input_layer.T))


    #run nerual net
    def predict(self, stimuli, raw=False):
        #set input layer to stimuli
        self.input_layer = np.array(stimuli, ndmin=2).T
        #set hidden layer to input layer run through weights_0 and sigmoided
        self.hidden_layer = sigmoid(np.dot(self.weights_0, self.input_layer))
        #set output layer to hidden layer run through weights_1 and sigmoided
        self.output_layer = sigmoid(np.dot(self.weights_1, self.hidden_layer))

        #if raw is set to true, return whole last layer
        if raw:
            return self.output_layer
        #else return index of largest value in last layer
        else:
            return np.argmax(self.output_layer)

