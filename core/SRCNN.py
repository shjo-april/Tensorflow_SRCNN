# Copyright (C) 2020 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import tensorflow as tf

def SRCNN(input_var, option):
    
    with tf.variable_scope('SRCNN'):
        x = input_var / 255.

        x = tf.layers.conv2d(inputs = x, **option['conv1'])
        x = tf.nn.relu(x)

        x = tf.layers.conv2d(inputs = x, **option['conv2'])
        x = tf.nn.relu(x)

        x = tf.layers.conv2d(inputs = x, **option['conv3'])
        
        if option['use_sigmoid']:
            x = tf.math.sigmoid(x)

    return x

if __name__ == '__main__':
    input_var = tf.placeholder(tf.float32, [None, 32, 32, 3])

    option = {
        'use_sigmoid' : True,
        
        'conv1' : dict(filters = 64, kernel_size = (9, 9), strides = 1, padding = 'same', name = 'conv1'),
        'conv2' : dict(filters = 32, kernel_size = (1, 1), strides = 1, padding = 'same', name = 'conv2'),
        'conv3' : dict(filters = 3, kernel_size = (3, 3), strides = 1, padding = 'same', name = 'conv3'),
    }

    x = SRCNN(input_var, option)

    # Tensor("SRCNN/Sigmoid:0", shape=(?, 32, 32, 3), dtype=float32)
    print(x)

