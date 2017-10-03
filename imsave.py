from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

x = mnist.train.images
x = x[1].reshape([28, 28]);

import scipy.misc
scipy.misc.imsave('digit.jpg', x)