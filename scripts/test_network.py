import numpy as np
from scripts.network import NeuralNetwork
from scripts.train_mnist import *
import pickle


if __name__ == '__main__':
    datapath = '../trained_networks/'
    network_name_pkl = ''
    with open(datapath + network_name_pkl, 'rb') as fp:
        test_nn: NeuralNetwork = pickle.load(fp)
    test_nn.feedforward()