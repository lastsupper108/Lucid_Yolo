import tensorflow as tf
IMAGE_H = 416
IMAGE_W = 416
NUM_CHANNAL = 3
GRID_DIM = 13

NUM_ANCHOR = 3
NUM_CLASSES = 20 

#images_folder     = './VOC2007/JPEGImages'
annotation_folder = './VOC2007/Annotations'
path_to_filenames = './VOC2007/ImageSets/Main/'



categories = ['boat', 'person', 'motorbike', 'car', 'diningtable', 'tvmonitor', 'bottle', 'train', 'chair', 'sofa', 'cow', 'cat', 'aeroplane', 'bus', 'bird', 'bicycle', 'horse', 'pottedplant', 'dog', 'sheep']
categories2idx = {u:i for i, u in enumerate(categories)}



BATCH_SIZE = 2
LEARNING_RATE = 5e-4

SHUFFLE_BUFFER_SIZE = 10


feature_description = {
            'image/height': tf.io.FixedLenFeature([], tf.int64),
            'image/width': tf.io.FixedLenFeature([], tf.int64),
            'image/depth': tf.io.FixedLenFeature([], tf.int64),
            'image/object/class/label': tf.io.VarLenFeature(tf.int64),
            'image/object/class/text': tf.io.VarLenFeature(tf.string),
            'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/filename': tf.io.FixedLenFeature([], tf.string)
        }



