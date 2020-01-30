

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
model_path = './experiments/model/SRCNN_image=32x32_batch=256_optimizer=SGD_lr=0.0001/1880000.ckpt'

pb_dir = './'
pb_name = 'SRCNN.pb'

# build SRCNN
image_var = tf.placeholder(tf.float32, [None, 32, 32, 1], name = 'images')

predictions_op = SRCNN(image_var, {
    'use_sigmoid' : True,
    
    'conv1' : dict(filters = 64, kernel_size = (9, 9), strides = 1, padding = 'same', name = 'conv1'),
    'conv2' : dict(filters = 32, kernel_size = (1, 1), strides = 1, padding = 'same', name = 'conv2'),
    'conv3' : dict(filters = 1, kernel_size = (3, 3), strides = 1, padding = 'same', name = 'conv3'),
})

sess = tf.Session()

saver = tf.train.Saver()
saver.restore(sess, model_path)
    
gd = sess.graph.as_graph_def()
converted_graph_def = graph_util.convert_variables_to_constants(sess, gd, ['SRCNN/outputs'])
tf.train.write_graph(converted_graph_def, pb_dir, pb_name, as_text=False)
print('freeze graph save complete')

