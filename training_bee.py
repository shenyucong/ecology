from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import os.path
import time
import tool

import tensorflow as tf


def deepnn(images):
  """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(images, W_conv1) + b_conv1)
    tf.summary.histogram("weights", W_conv1)
    tf.summary.histogram("biases", b_conv1)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    tf.summary.histogram("weights", W_conv2)
    tf.summary.histogram("biases", b_conv2)

  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    tf.summary.histogram("weights", W_fc1)
    tf.summary.histogram("biases", b_fc1)
    tf.summary.histogram("fc1/relu", h_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, 18])
    b_fc2 = bias_variable([18])

    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
    tf.summary.histogram("weights", W_fc2)
    tf.summary.histogram("biases", b_fc2)
  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name = "W")


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name = "B")

# Import data
tfrecords_file_train = 'bees_train.tfrecords'
tfrecords_file_test = 'bees_test.tfrecords'
train_dir = '/Users/chenyucong/Desktop/research/ecology/'
train_log_dir = '/Users/chenyucong/Desktop/research/ecology/log/'

filename = os.path.join(train_dir, tfrecords_file_train)
#filename_test = os.path.join(train_dir, tfrecords_file_test)
with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer([filename])
    #filename_queue_test = tf.train.string_input_producer([filename_test], num_epochs = 3)
    images, label = tool.read_and_decode(filename_queue)
    #images_test, label_test = read_and_decode(filename_queue_test)

    images_batch, label_batch = tf.train.shuffle_batch([images, label], batch_size=25, num_threads=1,capacity=1000 + 3 * 25, min_after_dequeue = 1000)
    #images_test_batch, label_test_batch = tf.train.batch([images_test, label_test], batch_size = 125, num_threads = 64, capacity = 1000+3*15)

#filename_test = os.path.join(train_dir, tfrecords_file_train)
#images_test, label_test = read_and_decode(filename_test)

x = tf.placeholder(tf.float32, [None, 28, 28, 1], name = "x")
tf.summary.image('input', x, 3)

  # Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 18], name = "labels")

  # Build the graph for the deep net
y_conv, keep_prob = deepnn(x)

with tf.name_scope('loss'):
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy, name = "loss")
  tf.summary.scalar("loss", cross_entropy)

with tf.name_scope('adam_optimizer'):
  train_step = tf.train.AdamOptimizer(1e-6).minimize(cross_entropy)

with tf.name_scope('accuracy'):
  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
  correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)
  tf.summary.scalar("accuracy", accuracy)

summ = tf.summary.merge_all()

#graph_location = tempfile.mkdtemp()
#print('Saving graph to: %s' % graph_location)
#train_writer = tf.summary.FileWriter(graph_location)
#train_writer.add_graph(tf.get_default_graph())

saver = tf.train.Saver(tf.global_variables())
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)
  tra_summary_writer = tf.summary.FileWriter(train_log_dir)
  tra_summary_writer.add_graph(sess.graph)
  try:
      for i in range(1000):
          images, label = sess.run([images_batch, label_batch])
          #images_test, label_test = sess.run([images_test_batch, label_test_batch])
          #print(images, label)
          #images_test, label_test = sess.run([images_test, label_test])
          if i % 100 == 0:
              train_accuracy = accuracy.eval(feed_dict={x: images, y_: label, keep_prob: 1.0})
              print('step %d, training accuracy %.4f' % (i, train_accuracy))
              #test_accuracy = accuracy.eval(feed_dict={x: images_test, y_: label_test, keep_prob: 1.0})
          _, tra_loss, summary = sess.run([train_step, cross_entropy, summ], feed_dict={x: images, y_: label})
          tra_summary_writer.add_summary(summary, i)
          if i % 100 == 0:
              print('step %d, loss %.4f' %(i, tra_loss))
          if i % 2000 == 0 or (i + 1) == 20000:
              checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
              saver.save(sess, checkpoint_path, global_step=i)
  except tf.errors.OutOfRangeError:
      print('Done training -- epoch limit reached')
  finally:
      coord.request_stop()

    #print('test accuracy %.4f' % accuracy.eval(feed_dict={
    #    x: images_test, y_: label_test, keep_prob: 1.0}))

  coord.request_stop()
  coord.join(threads)
