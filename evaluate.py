import os
import os.path

import numpy as np
import tensorflow as tf
import training_bee
import math
import tool

tfrecords_file_test = 'bees_test.tfrecords'
train_dir = '/Users/chenyucong/Desktop/research/ecology'
log_dir = ''

filename = os.path.join(train_dir, tfrecords_file_test)
with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer([filename])
    images_test, label_test = training_bee.read_and_decode(filename_queue)

    images_test_batch, label_test_batch = tf.train.batch([images_test, label_test], batch_size = 15, num_threads=1, capacity=1000+3*25)

y_conv, keep_prob = tool.deepnn(images_test)
correct = tool.num_correct_prediction(y_conv, label_test)
saver = tf.train.Saver(tf.global_variables())
with tf.Session() as sess:
    print("Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(log_dir)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Loading success, global_step is %s' % global_step)
    else:
        print('No checkpoint file found')
        return

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess, coord = coord)
    try:
        print('\nEvaluating......')
        num_step = int(math.floor(125 / 15))
        num_sample = num_step*15
        step = 0
        total_correct = 0
        while step < num_step and not coord.should_stop():
            batch_correct = sess.run(correct)
            total_correct += np.sum(batch_correct)
            step += 1
        print('Total testing samples: 125')
        print('Total correct predictions: %d' %total_correct)
        print('Average accuracy: %.2f%%' %(100*total_correct/num_sample))
    except Exception as e:
        coord.request_stop(e)
    finally:
        coord.request_stop()
        coord.join(threads)
