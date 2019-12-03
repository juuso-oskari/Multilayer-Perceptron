from scripts.network import NeuralNetwork
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

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

    train_data = np.array(list(zip(X_std, Y_train[:, 1])))

    test_data = np.array(list(zip(X_test_std, Y_test[:, 1])))

    nn = NeuralNetwork([16, 20, 20, 1])
    # shuffle the data to get batches to represent the whole data better
    np.random.shuffle(train_data)

    sample_amount = train_data.shape[0]
    batch_size = 20

    # some data to know when to end training
    continue_training = True
    learning_continues_check = 0
    threshold = 2
    tick = 0
    achieved_mse = 100
    set_init = True

    while continue_training:
        """for batch in np.split(np.array(train_data), sample_amount / batch_size, axis=0):
            nn.train_databatch(batch, 3.0)"""
        nn.train_databatch(train_data, 3.0)
        tick += 1
        if tick == 5:
            mse = mean_squared_error(Y_train[:, 1], [nn.feedforward(x) for x in X_std])
            print(mse)
            if set_init:
                achieved_mse = mse
                set_init = False

            if mse < achieved_mse:
                achieved_mse = mse
                learning_continues_check = 0
            else:
                learning_continues_check += 1

            tick = 0

        if learning_continues_check > 5 or achieved_mse < 0.004:
            continue_training = False

    """saving the trained network with filename network_<<achieved accuracy>>.pkl"""
    with open('../trained_networks/' + 'network_achieved_mse_' + str(0.004) + '.pkl', 'wb') as fp:
        pickle.dump(nn, fp)
