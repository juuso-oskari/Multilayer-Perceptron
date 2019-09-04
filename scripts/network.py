import numpy as np


class NeuralNetwork:
    # for example new NeuralNetwork(4,[8,8],2): 4 inputs, 2x 8 node hidden layers, 2 node output
    def __init__(self, input_layer, hidden_layers, output_layer):
        self.input_layer = input_layer
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer
        # during the feedforward, we save both the nodes and the inputs (the zs') before sigmoid function
        # for backpropagation purposes
        self.activations = []
        self.zs = []
        # initializations of weights and biases
        self.weights = self.initWeights()
        self.biases = self.initBiases()

    def initWeights(self):
        weights = []
        # first lets initialize the weights from input to first hidden layer
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
            bv = np.random.rand(self.hidden_layers[i])  # generates one bias-vector
            biases.append(bv)
        bv_o = np.random.rand(self.output_layer)
        biases.append(bv_o)
        return biases
    @staticmethod
    def cost_prime(outputs, targets):
        return 2*(outputs - targets)

    @staticmethod
    def activationFunction(x):
        return np.exp(x)/(np.exp(x)+1)  # this is basic sigmoid function

    @staticmethod
    def s_prime(x):
        return np.exp(x)/(np.exp(x)+1) - np.exp(2*x)/np.power((np.exp(x)+1), 2)

    def feedForward(self, input):
        self.activations = []
        self.zs = []  # empty the node and bias values from previous feedforward
        if len(input) == self.input_layer:
            a = np.array(input)
            self.activations.append(a)
            bias_index = 0
            for w in self.weights:
                w_a = np.matmul(w, a)  # W*A^(L-1)  ^:upper index, not power
                z = np.subtract(w_a, self.biases[bias_index])  # W*A^(L-1)-B^L
                self.zs.append(z)
                a = np.array(list(map(self.activationFunction, z)))  # sigmoid(Z)
                self.activations.append(a)
                bias_index += 1
            # return output
            return a
        else:
            print('wrong input size')
            return None

    def backpropagate(self, outputs, targets, eta):  # todo only calculate do not adjust, instead return deltas
        errors = self.cost_prime(outputs, targets)  # Gc/Ga
        for i in range(0, len(self.weights) - 1):
            c_l = len(self.hidden_layers) - i  # index for current layer (indexing starts at zero)
            # calculate gradients Ga/Gz*Gc/Ga
            gradients = np.array(list(map(self.s_prime, self.zs[c_l])))
            gradients = np.multiply(gradients, errors)
            # calculate bias deltas and adjust biases
            b_deltas = gradients
            self.biases[c_l] = np.subtract(self.biases[c_l], eta*b_deltas)
            # calculate weight deltas (i.e the amount we have to change the weights)
            # this might look stupid, but its basically a dot product with activations transposed.
            # numpy just made it a misery
            w_deltas = np.dot(gradients.reshape(-1, 1), self.activations[c_l].reshape((1, -1)))
            # adjust weights
            self.weights[c_l] = np.subtract(self.weights[c_l], eta*w_deltas)
            # and lets calculate the errors for the next layer
            errors = np.matmul(np.transpose(self.weights[c_l]), gradients)

    def train(self, inputs, targets, eta):
        for i in range(1, 100):
            outputs = self.feedForward(inputs)
            self.backpropagate(outputs, targets, eta)


if __name__ == '__main__':

    """output = nn.feedForward([1, 2, 3, 4])
    print("output with untrained network: ")
    print(output)"""

    error_count = 0
    for i in range(1, 100):
        nn = NeuralNetwork(4, [3, 3, 3], 2)
        nn.train([1, 2, 3, 4], [1, 0], 0.7)
        trained_output = nn.feedForward([1, 2, 3, 4])
        if trained_output[0] < trained_output[1]:
            print("error")
            error_count += 1
    trained_output = nn.feedForward([1, 2, 3, 4])
    print(trained_output)
    print("mistakes made (%): " + str(error_count / 100))
