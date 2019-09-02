import numpy as np

class NeuralNetwork:

    def __init__(self, input_layer, hidden_layers, output_layer):   # for example new NeuralNetwork(4,[8,8],2): 4 inputs, 2x 8 node hidden layers, 2 node output
        self.input_layer = input_layer
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer
        # initializations
        self.weights = self.initWeights()
        self.biases = self.initBiases()

    def initWeights(self):
        weights = []
        # first lets initialize the weigths from input to first hidden layer
        w_ih = np.random.rand(self.hidden_layers[0], self.input_layer)
        weights.append(w_ih)
        # then go over the hidden layers weights
        for i in range(1, len(self.hidden_layers)):
            w_hh = np.random.rand(self.hidden_layers[i], self.hidden_layers[i-1])
            weights.append(w_hh)
        # and at last from last hidden layer to output
        w_ho = np.random.rand(self.output_layer, self.hidden_layers[len(self.hidden_layers)-1])
        weights.append(w_ho)
        return weights

    def initBiases(self):
        biases = []
        for i in range(0, len(self.hidden_layers)):
            bv = np.random.rand(self.hidden_layers[i]) #generates one biasvector
            biases.append(bv)
        bv_o = np.random.rand(self.output_layer)
        biases.append(bv_o)
        return biases

    def activationFunction(self, x):
        return np.exp(x)/(np.exp(x)+1)

    def feedForward(self, input):
        if len(input) == self.input_layer:
            a = input
            bias_index = 0
            for w in self.weights:
                a = np.matmul(w, a)
                a = np.subtract(a, self.biases[bias_index])
                a = list(map(self.activationFunction, a))
                bias_index +=1
            #return output
            return a
        else:
            print('wrong input size')
            return None
    def derivateToWeight(self, l, i, j):
        total_layers = 2 + len(self.hidden_layers)
        iterations = total_layers - l
        d = 1
        i = 0
        while i<iterations:
           
           i+=1

        pass
if __name__ == '__main__' :
    nn = NeuralNetwork(4, [10, 2], 2)
    output = nn.feedForward([1,2,2,3])
    print(output)
