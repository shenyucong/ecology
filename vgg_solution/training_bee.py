from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import os.path
import time
import tool
import vgg

import tensorflow as tf


# Import data
tfrecords_file_train = 'bees_train.tfrecords'
tfrecords_file_test = 'bees_test.tfrecords'
train_dir = '/Users/chenyucong/Desktop/research/ecology/vgg_solution/'
train_log_dir = '/Users/chenyucong/Desktop/research/ecology/vgg_solution/log/'

filename = os.path.join(train_dir, tfrecords_file_train)
#filename_test = os.path.join(train_dir, tfrecords_file_test)i
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
y_ = tf.placeholder(tf.float32, [None, 19], name = "labels")

  # Build the graph for the deep net
y_conv = vgg.VGG16(x, 19, True)

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
  tool.load_with_skip(pre_trained_weights, sess, ['fc6','fc7','fc8'])
  sess.run(tf.global_variables_initializer())
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)
  tra_summary_writer = tf.summary.FileWriter(train_log_dir)
  tra_summary_writer.add_graph(sess.graph)
  try:
      for i in range(15000):
          images, label = sess.run([images_batch, label_batch])
          #images_test, label_test = sess.run([images_test_batch, label_test_batch])
          #print(images, label)
          #images_test, label_test = sess.run([images_test, label_test])
          if i % 100 == 0:
              train_accuracy = accuracy.eval(feed_dict={x: images, y_: label, keep_prob: 1.0})
              #test_accuracy = accuracy.eval(feed_dict={x: images_test, y_: label_test, keep_prob: 1.0})
          _, tra_loss, summary = sess.run([train_step, cross_entropy, summ], feed_dict={x: images, y_: label})
          tra_summary_writer.add_summary(summary, i)
          if i % 100 == 0:
              print('step %d, loss %.4f, trainig accuracy %.4f' %(i, tra_loss, train_accuracy))
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
