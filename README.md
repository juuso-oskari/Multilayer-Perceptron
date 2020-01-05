# MultilayerPerceptron
My implementation of mlp neural network, tested with Mnist and weather data sets.

RESULTS OF THE TESTING:

Mnist is a popular train and dataset of handwritten digits. I was able to achieve 94% accuracy with only about a minute training, which is quite good. The dataset is available on http://yann.lecun.com/exdb/mnist/.
The weather data set was concerning a school course, where the aim was to predict humidity of the day if otherkinds of parameters like temperature and visibility is known. Interesting remark from this was that neural network managed
to perform much better than other predictions made with Linear Regression and KNN (Mean square error with nn went well below 0.04 as with lr it was 2.4 and KNN 7.2).

ABOUT SCRIPTS:

Mnistloader.py file is a loader for the mnist dataset. Comments in the code file offer instructions.
Network.py file gives my implementation of the mlp network.
The other files are my implementations of training and testing.
