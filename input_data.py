import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import skimage.io as io
import random
from skimage.transform import resize
from PIL import Image

def get_file(file_dir):
    '''
    Args:
        file_dir: file directory
    Returns:
        images: image directories, list, string
        labels: label, list, int
    '''

    images = []
    temp = []
    for root, sub_folders, files in os.walk(file_dir):
        #image directories
        for name in files:
            images.append(os.path.join(root, name))
        #get 10 sub-folder names
        for name in sub_folders:
            temp.append(os.path.join(root, name))

    #assign 18 labels on the folder names
    labels = []
    for one_folder in temp:
        n_img = len(os.listdir(one_folder))
        letter = one_folder.split('/')[-1]

        if letter == 'agapostemonvirescens':
            labels = np.append(labels, n_img*[0])
        elif letter == 'augochlorapura':
            labels = np.append(labels, n_img*[1])
        elif letter == 'augochlorellastriata':
            labels = np.append(labels, n_img*[2])
        elif letter == 'bombusimpatiens':
            labels = np.append(labels, n_img*[3])
        elif letter == 'ceratinacalcarata':
            labels = np.append(labels, n_img*[4])
        elif letter == 'ceratinadupla':
            labels = np.append(labels, n_img*[5])
        elif letter == 'ceratinametallica':
            labels = np.append(labels, n_img*[6])
        elif letter == 'dialictusbruneri':
            labels = np.append(labels, n_img*[7])
        elif letter == 'dialictusillinoensis':
            labels = np.append(labels, n_img*[8])
        elif letter == 'dialictusimitatus':
            labels = np.append(labels, n_img*[9])
        elif letter == 'dialictusrohweri':
            labels = np.append(labels, n_img*[10])
        elif letter == 'halictusconfusus':
            labels = np.append(labels, n_img*[11])
        elif letter == 'halictusligatus':
            labels = np.append(labels, n_img*[12])
        elif letter == 'osmiaatriventis':
            labels = np.append(labels, n_img*[13])
        elif letter == 'osmiabucephala':
            labels = np.append(labels, n_img*[14])
        elif letter == 'osmiacornifrons':
            labels = np.append(labels, n_img*[15])
        elif letter == 'osmiageorgica':
            labels = np.append(labels, n_img*[16])
        elif letter == 'osmialignaria':
            labels = np.append(labels, n_img*[17])
        elif letter == 'osmiapumila':
            labels = np.append(labels, n_img*[18])
    #shuffle
    #print(temp)
    #temp = temp.transpose()
    #np.random.shuffle(temp)
    #print(temp)

    image_list = list(images)
    labels = list(labels)

    return image_list, labels

def int64_feature(value):
    '''Wrapper for inserting int64 features into Example proto.'''
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list = tf.train.Int64List(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def convert_to_tfrecord(images, labels, save_dir, name):
    '''
    Args:
        images: list of image directories, string type
        labels: list of labels, int type
        save_dir: the drectory to save tfrecord file, e.g.: '/home/folder1/'
        name: the name of tfrecord file, string type, e.g.: 'train'

    Return:
        no return
    '''

    filename = os.path.join(save_dir, name + '.tfrecords')
    images = images[0:len(images)-1]
    n_samples = len(labels)

    if np.shape(images)[0] != n_samples:
        raise ValueError('Images size %d does not match label size %d.' %(len(images), n_samples))

    writer = tf.python_io.TFRecordWriter(filename)
    print('\nTransform start......')
    i = 0
    j = 0
    for i in np.arange(0, n_samples):
        try:
            image = io.imread(images[i], as_grey=True)
            #limage = resize(image, (28, 28))
            image_raw = image.tostring()
            label = int(labels[i])
            height, width = image.shape
            example = tf.train.Example(features = tf.train.Features(feature={
                'height': int64_feature(height),
                'width': int64_feature(width),
                'label':int64_feature(label),
                'image_raw': bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())
        except IOError as e:
            print('Could not read:', images[i])
            print('error: %s' %e)
            print('Skip it\n')
    writer.close()
    print('Transform done!\n')

def read_and_decode(tfrecords_file, batch_size):
    '''
    Args:
        tfrecords_file: the directory of tfrecord file
        batch_size: number of images in each batch
    Returns:
        image: 4D tensor - [batch_size, with, height, channel]
        label:1D tensor - [batch_size]
    '''

    #make an input queue from the tfrecord file
    filename_queue = tf.train.string_input_producer([tfrecords_file])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    img_features = tf.parse_single_example(
        serialized_example,
        features={
            'height':tf.FixedLenFeature([], tf.int64),
            'width' :tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
        })
    height = tf.cast(img_features['height'], tf.int32)
    width = tf.cast(img_features['width'], tf.int32)
    image = tf.decode_raw(img_features['image_raw'], tf.uint8)



    image = tf.reshape(image, [height, width])
    image = tf.expand_dims(image, -1)
    image = tf.image.resize_images(image, (28, 28))
    label = tf.cast(img_features['label'], tf.int32)
    image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 64,
                                                capacity = 2000,
                                                min_after_dequeue = 1000)
    #image_batch = tf.expand_dims(image_batch, -1)
    return image_batch, tf.reshape(label_batch, [batch_size])

train_dir = '/Users/chenyucong/Desktop/research/ecology/data/train'
test_dir = '/Users/chenyucong/Desktop/research/ecology/data/test'
save_dir = '/Users/chenyucong/Desktop/research/ecology/'

name_train = 'bees_train'
name_test = 'bees_test'
images_train, labels_train = get_file(train_dir)
images_test, labels_test = get_file(test_dir)
convert_to_tfrecord(images_train, labels_train, save_dir, name_train)
convert_to_tfrecord(images_test, labels_test, save_dir, name_test)

tfrecords_file_train = '/Users/chenyucong/Desktop/research/ecology/bees_train.tfrecords'
#tfrecords_file_test = '/Users/chenyucong/Desktop/research/ecology/bees_test.tfrecords'
image_batch, label_batch = read_and_decode(tfrecords_file_train, batch_size = 25)
print(image_batch)
#image_test, label_test = read_and_decode(tfrecords_file_train, batch_size = Batch_size_test)
#print(image_batch, label_batch)
#print(image_test, label_test)
#item = list()
#for i in range(772):
#    item.append(i)
#testlist=random.sample(range(1,772),100)
#trainlist = list(set(item).difference(set(testlist)))
#image_train = list()
#image_test = list()
#label_train = list()
#label_test = list()
#for i in testlist:
#    image_test = image_test.append(image_batch[i,:,:])
#    label_test = label_test.append(label_batch[i])

#for j in trainlist:
#    image_train = image_train.append(image_batch[j,:,:])
#    label_test = label_test.append(label_batch[i])

#image_batch = tf.expand_dims(image_batch, -1)
#image_test = tf.expand_dims(image_test, -1)
#image_batch = tf.image.resize_images(image_batch, [28,28])
#image_test = tf.image.resize_images(image_test, [28,28])
#label_batch = tf.one_hot(indices=tf.cast(label_batch, tf.int32),depth = 18)
#label_test = tf.one_hot(indices=tf.cast(label_test, tf.int32),depth = 18)
#image_test = tf.expand_dims(image_test, -1)
#onehot_label_test = tf.one_hot(indices = tf.cast(label_test, tf.int32),depth = 18)
def plot_images(images, labels):
    for i in np.arange(0, 25):
        plt.subplot(5, 5, i + 1)
        plt.axis('off')
        plt.title(chr(ord('A') + labels[i] - 1), fontsize = 14)
        plt.subplots_adjust(top=1.5)
        plt.imshow(images[i])
    plt.show()
with tf.Session() as sess:

    i = 0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        while not coord.should_stop() and i<1:
            #just plot one batch size
            image, label = sess.run([image_batch, label_batch])
            print(image)
            plot_images(image, label)
            i+=1

    except tf.errors.OutOFRangeError:
        print('done!\n')
    finally:
        coord.request_stop()
    coord.join(threads)
