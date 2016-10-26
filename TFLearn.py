# http://tflearn.org/installation/#tflearn-installation
import tflearn

data, labels, test_data, test_labels = tflearn.datasets.mnist.load_data(one_hot=True)

input_layer = tflearn.input_data(shape=[None, 784])
softmax = tflearn.fully_connected(input_layer, 10, activation='softmax')

net = tflearn.regression(softmax, optimizer='sgd', loss='categorical_crossentropy')

model = tflearn.DNN(net)
model.fit(data, labels, validation_set=(test_data, test_labels), show_metric=True)
