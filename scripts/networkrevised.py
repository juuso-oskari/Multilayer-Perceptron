import numpy as np


class NeuralNetwork:
    # for example new NeuralNetwork([4, 8, 8, 2]): 4 inputs, 2x 8 node hidden layers, 2 node output
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.num_layers = len(self.layer_sizes)
        # initialization of biases and weights, np.random.randn: random numbers between 0,1 according to gaussian db
        self.biases = [np.random.randn(l) for l in self.layer_sizes[1:self.num_layers]]
        self.weights = [np.random.randn(l2, l1) for l1, l2 in zip(self.layer_sizes[0:self.num_layers - 1],
                                                                  self.layer_sizes[1:self.num_layers])]
        """during the feedforward of the input, we save both the activations and the inputs to layers (the zs)
        for backpropagation purposes"""
        self.activations = []
        self.zs = []

    @staticmethod
    def cost_prime(outputs, targets):
        return 2*(outputs - targets)  # from cost_function = (outputs - targets)^2

    @staticmethod
    def activation_function(x):
        return np.exp(x)/(np.exp(x)+1)  # sigmoid function

    @staticmethod
    def s_prime(x):
        return np.exp(x)/(np.exp(x)+1) - np.exp(2*x)/np.power((np.exp(x)+1), 2)

    def feedforward(self, input):
        """empty the node and bias values from previous feedforward"""
        self.activations = []
        self.zs = []
        if len(input) == self.layer_sizes[0]:
            a = np.array(input)
            self.activations.append(a)
            for w, b in zip(self.weights, self.biases):
                z = np.dot(w, a) - b
                self.zs.append(z)
                a = self.activation_function(z)  # sigmoid(Z)
                self.activations.append(a)
            # return output
            return a
        else:
            raise Exception('input size should be {}, input size was {}'.format(self.layer_sizes[0], len(input)))

    def backpropagate(self, outputs, targets, eta):  # todo only calculate do not adjust, instead return deltas
        errors = self.cost_prime(outputs, targets)
        for i in range(1, self.num_layers - 1):
            # calculate gradients Ga/Gz*Gc/Ga
            gradients = self.s_prime(self.zs[-i])  # np.array(list(map(self.s_prime, self.zs[c_l])))

            gradients = np.multiply(gradients, errors)

            # calculate bias deltas and adjust biases
            b_deltas = gradients
            self.biases[-i] = np.subtract(self.biases[-i], eta*b_deltas)
            # calculate weight deltas (i.e the amount we have to change the weights)
            # this might look stupid, but its basically a dot product with activations transposed.
            # numpy just made it a misery
            w_deltas = np.dot(gradients.reshape(-1, 1), self.activations[-i-1].reshape((1, -1)))
            # adjust weights
            self.weights[-i] = np.subtract(self.weights[-i], eta*w_deltas)
            # and lets calculate the errors for the next layer
            errors = np.dot(np.transpose(self.weights[-i]), gradients)

    def train(self, inputs, targets, eta):
        for i in range(1, 100):
            outputs = self.feedforward(inputs)
            self.backpropagate(outputs, targets, eta)


if __name__ == '__main__':

    """output = nn.feedForward([1, 2, 3, 4])
    print("output with untrained network: ")
    print(output)"""

    error_count = 0
    for i in range(1, 100):
        nn = NeuralNetwork([4, 3, 3, 3, 2])
        nn.train([1, 2, 3, 4], [1, 0], 0.7)
        trained_output = nn.feedforward([1, 2, 3, 4])
        if trained_output[0] < trained_output[1]:
            print("error")
            error_count += 1
    trained_output = nn.feedforward([1, 2, 3, 4])
    print(trained_output)
    print("mistakes made (%): " + str(error_count / 100))
