import os
import os.path

import numpy as np
import tensorflow as tf
import math
import tool
import enet
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

tfrecords_file_test = 'bees_test.tfrecords'
train_dir = '/data/shaobo/Data/'
log_dir = '/data/shaobo/Data/enetlog0325/'

filename = os.path.join(train_dir, tfrecords_file_test)
with tf.name_scope('input'):
    images_test, label_test = tool.read_and_decode(filename, 100)

    #images_test_batch, label_test_batch = tf.train.batch([images_test, label_test], batch_size = 200, num_threads=64, capacity=1000+3*200)

y_conv = enet.enet(images_test, 19)
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

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess, coord = coord)
    try:
        print('\nEvaluating......')
        num_step = int(math.floor(3600 / 100))
        num_sample = num_step*100
        step = 0
        total_correct = 0
        while step < num_step and not coord.should_stop():
            batch_correct = sess.run(correct)
            total_correct += np.sum(batch_correct)
            step += 1
        print('Total testing samples: 3600')
        print('Total correct predictions: %d' %total_correct)
        print('Average accuracy: %.2f%%' %(100*total_correct/num_sample))
    except Exception as e:
        coord.request_stop(e)
    finally:
        coord.request_stop()
        coord.join(threads)
