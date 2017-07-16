from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

import tensorflow as tf
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(
  tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
with tf.name_scope("accuracy"):
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar("accuracy", accuracy)

sess = tf.InteractiveSession()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("/tmp/mnist_logs",  sess.graph)
tf.global_variables_initializer().run()

for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y_: batch_ys})
  writer.add_summary(summary, i)

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))