import numpy as np
import pickle


class NeuralNetwork:
    # for example new NeuralNetwork([4, 8, 8, 2]): 4 inputs, 2x 8 node hidden layers, 2 node output
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.num_layers = len(self.layer_sizes)
        # initialization of biases and weights, np.random.randn: random numbers according to gaussian db
        self.biases = [np.random.randn(l) for l in self.layer_sizes[1:self.num_layers]]
        self.weights = [np.random.randn(l2, l1) for l1, l2 in zip(self.layer_sizes[0:self.num_layers - 1],
                                                                  self.layer_sizes[1:self.num_layers])]
        """during the feedforward of the input, we save both the activations and the inputs to layers (the zs)
        for backpropagation purposes"""
        self.activations = []
        self.zs = []

    @staticmethod
    def cost_function(outputs, targets):
        return np.power(outputs - targets, 2)

    @staticmethod
    def cost_prime(outputs, targets):
        return 2 * (outputs - targets)  # derivative of cost function with the output

    @staticmethod
    def activation_function(x):
        return np.exp(x)/(np.exp(x)+1)  # sigmoid function

    @staticmethod
    def activation_prime(x):
        """def der(i):
            if i > 0.0:
                return 1.0
            else:
                return 0.0

        return np.array(list(map(der, x)))
    #"""
        return np.exp(x)/(np.exp(x)+1) - np.exp(2*x)/np.power((np.exp(x)+1), 2)  # derivative of sigmoid

    def feedforward(self, input):
        """empty the activation and zs values from previous feedforward"""
        self.activations = []
        self.zs = []  # inputs to activation function
        if len(input) == self.layer_sizes[0]:
            a = np.array(input)
            self.activations.append(a)
            for w, b in zip(self.weights, self.biases):
                z = np.dot(w, a) + b
                self.zs.append(z)
                a = self.activation_function(z)  # sigmoid(z)
                self.activations.append(a)
            # return output
            return a
        else:
            raise Exception('input-vector size should be {}, input-vector size was {}'.format(self.layer_sizes[0],
                                                                                              len(input)))

    def backpropagate(self, outputs, targets):
        error_gradients = self.cost_prime(outputs, targets)  # delC / delA
        # for collection of the deltas
        nabla_b = [np.zeros(l) for l in self.layer_sizes[1:self.num_layers]]
        nabla_w = [np.zeros((l2, l1)) for l1, l2 in zip(self.layer_sizes[0:self.num_layers - 1],
                                                        self.layer_sizes[1:self.num_layers])]
        for i in range(1, self.num_layers):
            gradients = error_gradients * self.activation_prime(self.zs[-i])  # delC / delA * delA / delZ
            b_deltas = gradients  # since delZ / delB = 1
            nabla_b[-i] = b_deltas
            w_deltas = np.dot(gradients.reshape((-1, 1)), (self.activations[-i-1].reshape((-1, 1)).transpose()))
            nabla_w[-i] = w_deltas
            error_gradients = np.dot(np.transpose(self.weights[-i]), gradients)
        return nabla_b, nabla_w

    def train(self, inputs, targets):
        if len(targets) == self.layer_sizes[-1]:
            outputs = self.feedforward(inputs)
            return self.backpropagate(outputs, targets)
        else:
            raise Exception('target-vector size: {} did not match the networks output-vector size: {}'
                            .format(len(targets), self.layer_sizes[-1]))

    def train_databatch(self, databatch, eta):
        nabla_matrix = [self.train(inputs, targets) for inputs, targets in databatch]
        delta_b = np.average([x[0] for x in nabla_matrix], axis=0)
        delta_w = np.average([x[1] for x in nabla_matrix], axis=0)
        self.biases = [b - eta * d_b for b, d_b in zip(self.biases, delta_b)]
        self.weights = [w - eta * d_w for w, d_w in zip(self.weights, delta_w)]

    def validate(self, validation_data):
        test_amount = len(validation_data)
        accuracy = 1.0
        for test in validation_data:
            input = test[0]
            t_output = test[1]
            output = self.feedforward(input)
            if np.argmax(output) != np.argmax(t_output):
                accuracy -= 1.0 / test_amount
        return accuracy
