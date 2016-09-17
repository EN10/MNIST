from PIL import Image,ImageFilter
im = Image.open('digit.bmp')
im.save('out.png')

tv = list(im.getdata()) #get pixel values
#normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
imvalue = [ (255-x)*1.0/255.0 for x in tv] 

import tensorflow as tf
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])

init = tf.initialize_all_variables()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, "model.ckpt")
  
    prediction=tf.argmax(y,1)
    #first value in list
    print (prediction.eval(feed_dict={x: [imvalue]}, session=sess)[0])