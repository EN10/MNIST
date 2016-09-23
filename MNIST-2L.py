from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

import tensorflow as tf
x = tf.placeholder(tf.float32, [None, 784])
N = 100
w2 = tf.Variable(tf.truncated_normal([784, N], stddev=0.1))
b2 = tf.Variable(tf.zeros([N]))
y2 = tf.nn.relu(tf.matmul(x, w2) + b2)
w = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(y2, w) + b)
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(1000):
  batch_x, batch_y = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))