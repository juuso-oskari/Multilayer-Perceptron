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

    def train_databatch(self, inputs_matrix, targets_matrix, eta):
        nabla_matrix = [self.train(inputs, targets) for inputs, targets in zip(inputs_matrix, targets_matrix)]
        delta_b = np.average([x[0] for x in nabla_matrix], axis=0)
        delta_w = np.average([x[1] for x in nabla_matrix], axis=0)
        self.biases = [b - eta * d_b for b, d_b in zip(self.biases, delta_b)]
        self.weights = [w - eta * d_w for w, d_w in zip(self.weights, delta_w)]

    def validate(self, validation_data):
        test_amount = len(validation_data)
        accuracy = 1.0
        for test in validation_data:
            for input, target_output in test:
                output = self.feedforward(input)
                if output.index(max(output)) != target_output.index(max(target_output)):
                    accuracy -= 1.0 / test_amount
        return accuracy


if __name__ == '__main__':
    nn = NeuralNetwork([784, 18, 18, 10])
    """loading the MNIST-dataset from pickle file MNISTData.pkl to a key-value dictionary data_dict"""
    datapath = '../data/MNISTData/'
    with open(datapath + 'MNISTData.pkl', 'rb') as fp:
        data_dict = pickle.load(fp)
    lr = np.arange(10)
    # transform labels into one hot representation, "1" becomes [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    train_labels = (lr == data_dict['train_labels']).astype(np.float)
    test_labels = (lr == data_dict["test_labels"]).astype(np.float)
    # we don't want zeroes and ones in the labels neither:
    train_labels[train_labels == 0] = 0.01
    train_labels[train_labels == 1] = 0.99
    test_labels[test_labels == 0] = 0.01
    test_labels[test_labels == 1] = 0.99

    train_data = zip(data_dict['train_images'], train_labels)
    test_data = zip(data_dict["test_images"], test_labels)
    """shuffle the data for the batch to represent the whole data better"""
    np.random.shuffle(train_data)
    sample_amount, i_h, i_w = data_dict["train_images"].shape
    batch_size = 100
    continue_training = True
    while continue_training:
        for batch in np.split(np.array(train_data), sample_amount / batch_size, axis=0):
            for i_m, t_m in batch:
                nn.train_databatch(i_m.reshape((1, -1)), t_m, 3)
        accuracy = nn.validate(np.array(test_data))
        print(accuracy)
        if accuracy > 0.9:
            continue_training = False
