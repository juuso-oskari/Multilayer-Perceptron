import numpy as np
from scripts.network import NeuralNetwork
import pickle

"""BEFORE USING THIS FILE, MAKE SURE THAT YOU HAVE DOWNLOADED THE MNIST-DATASET BY RUNNING MNISTloader.py !!!
AFTER DOWNLOADING, THE DATAFILES (-ubyte ending) ALONGSIDE THE PICKLE-FILE (.pkl ending file), WHICH WE USE HERE
SHOULD BE FOUND FROM .../data/MNISTData/ FOLDER"""


def images_to_pixel_vectors(image_array):
    img_amount, w, h = image_array.shape
    return np.reshape(image_array, (img_amount, -1))


def one_hot_encode(label_number):
    lr = np.arange(10.0)
    for i in range(0, 10):

        if lr[i] == label_number:
            lr[i] = 0.99
        else:
            lr[i] = 0.01
    return lr


def train_mnist(network, treshold):
    nn = network
    """loading the MNIST-dataset from pickle file MNISTData.pkl to a key-value dictionary data_dict"""
    datapath = '../data/MNISTData/'
    with open(datapath + 'MNISTData.pkl', 'rb') as fp:
        data_dict = pickle.load(fp)
    """changing the scale of pixel values from 0-255 to 0.1-1"""
    fac = 0.99 / 255
    train_imgs = np.asfarray(data_dict["train_images"]) * fac + 0.01
    test_imgs = np.asfarray(data_dict["test_images"]) * fac + 0.01
    train_labels = np.asfarray(data_dict["train_labels"])
    test_labels = np.asfarray(data_dict["test_labels"])
    """transform labels into one hot representation, "1" becomes [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]"""
    train_labels = [one_hot_encode(lbl) for lbl in train_labels]
    test_labels = [one_hot_encode(lbl) for lbl in test_labels]
    """from 28x28 images to 784-pixel vectors"""
    train_imgs = images_to_pixel_vectors(train_imgs)
    test_imgs = images_to_pixel_vectors(test_imgs)
    train_data = np.array(list(zip(train_imgs, train_labels)))
    test_data = np.array(list(zip(test_imgs, test_labels)))
    """shuffle the data for the batch to represent the whole data better"""
    np.random.shuffle(train_data)
    sample_amount = train_data.shape[0]
    batch_size = 100
    continue_training = True
    achieved_accuracy = 0
    learning_continues_check = 0
    while continue_training:
        for batch in np.split(np.array(train_data), sample_amount / batch_size, axis=0):
            nn.train_databatch(batch, 3.0)
        accuracy = nn.validate(np.array(test_data))
        print(accuracy)
        if accuracy < achieved_accuracy:
            learning_continues_check += 1
        else:
            learning_continues_check = 0
        if accuracy > treshold or learning_continues_check > 4:
            continue_training = False
    """saving the trained network with filename network_<<achieved accuracy>>.pkl"""
    with open('../trained_networks/'+'network_max_accuracy_'+str(treshold)+'.pkl', 'wb') as fp:
        pickle.dump(nn, fp)


if __name__ == '__main__':
    test_network = NeuralNetwork([784, 50, 50, 10])
    train_mnist(test_network, 0.98)
