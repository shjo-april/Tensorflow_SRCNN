

import os
import cv2
import time

import numpy as np
import tensorflow as tf

from core.SRCNN import *

from tensorflow.python.platform import app
from tensorflow.python.summary import summary
from tensorflow.python.framework import graph_util

# define
model_path = './experiments/model/SRCNN_image=33x33_batch=128_optimizer=Momentum_lr=0.0001/310000.ckpt'

pb_dir = './test/'
pb_name = 'SRCNN.pb'

# build SRCNN
image_var = tf.placeholder(tf.float32, [None, 33, 33, 3], name = 'images')

predictions_op = SRCNN(image_var, {
    'use_sigmoid' : False,
    
    'conv1' : dict(filters = 64, kernel_size = (9, 9), strides = 1, padding = 'same', kernel_initializer = tf.contrib.layers.xavier_initializer(), name = 'conv1'),
    'conv2' : dict(filters = 32, kernel_size = (1, 1), strides = 1, padding = 'same', kernel_initializer = tf.contrib.layers.xavier_initializer(), name = 'conv2'),
    'conv3' : dict(filters = 3, kernel_size = (5, 5), strides = 1, padding = 'same', kernel_initializer = tf.contrib.layers.xavier_initializer(), name = 'conv3'),
})

sess = tf.Session()

saver = tf.train.Saver()
saver.restore(sess, model_path)

gd = sess.graph.as_graph_def()
converted_graph_def = graph_util.convert_variables_to_constants(sess, gd, ['SRCNN/outputs'])
tf.train.write_graph(converted_graph_def, pb_dir, pb_name, as_text=False)
print('freeze graph save complete')

