import numpy as np
from scripts.train_mnist import *
from scripts.network import NeuralNetwork
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

if __name__ == '__main__':


    X_train = pd.read_csv('../weather_data/weather_data_train.csv', index_col='datetime',
                          sep=';', decimal=',', infer_datetime_format=True).to_numpy()

    Y_train = pd.read_csv('../weather_data/weather_data_train_labels.csv', index_col='datetime',
                          sep=';', decimal=',', infer_datetime_format=True).to_numpy()

    X_test = pd.read_csv('../weather_data/weather_data_test.csv', index_col='datetime',
                         sep=';', decimal=',', infer_datetime_format=True).to_numpy()

    Y_test = pd.read_csv('../weather_data/weather_data_test_labels.csv', index_col='datetime',
                         sep=';', decimal=',', infer_datetime_format=True).to_numpy()

    # lets standardize our data first
    X_std = StandardScaler().fit_transform(X_train)
    Y_train[:, 1] = Y_train[:, 1] / 100
    X_test_std = StandardScaler().fit_transform(X_test)
    Y_test[:, 1] = Y_test[:, 1] / 100

    datapath = '../trained_networks/'
    network_name_pkl = 'network_achieved_mse_0.004.pkl'
    with open(datapath + network_name_pkl, 'rb') as fp:
        nn: NeuralNetwork = pickle.load(fp)
    mse = mean_squared_error(Y_test[:, 1], [nn.feedforward(x) for x in X_test_std])
    print(mse)
