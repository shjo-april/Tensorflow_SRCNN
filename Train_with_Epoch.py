# Copyright (C) 2020 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import os
import cv2
import glob
import time
import argparse

import numpy as np
import tensorflow as tf

from queue import Queue

from core.SRCNN import *

from utils.Utils import *

def parse_args():
    parser = argparse.ArgumentParser(description='SRCNN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--use_gpu', dest='use_gpu', help='use gpu', default='0', type=str)
    
    parser.add_argument('--color', dest='color', help='color', default=True, type=bool)
    parser.add_argument('--scale', dest='scale', help='scale', default=3, type=int)
    parser.add_argument('--stride', dest='stride', help='stride', default=14, type=int)
    
    parser.add_argument('--image_size', dest='image_size', help='image_size', default=32, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='batch_size', default=128, type=int)

    parser.add_argument('--optimizer', dest='optimizer', help='optimizer', default='SGD', type=str)
    parser.add_argument('--learning_rate', dest='learning_rate', help='learning_rate', default=1e-4, type=float)
    
    parser.add_argument('--save_epoch', dest='save_epoch', help='save_epoch', default=100, type=int)
    parser.add_argument('--max_epoch', dest='max_epoch', help='max_epoch', default=15000, type=int)

    return parser.parse_args()

##########################################################################################################
# 모든 파라미터를 준비하고, 중요한 정보는 저장합니다.
##########################################################################################################
args = vars(parse_args())

if args['color']:
    args['color_depth'] = 3
else:
    args['color_depth'] = 1

os.environ["CUDA_VISIBLE_DEVICES"] = args['use_gpu']

folder_name = 'SRCNN_image={}x{}_color={}_optimizer={}_lr={}_stride={}'.format(args['image_size'], args['image_size'], args['color'], args['optimizer'], args['learning_rate'], args['stride'])

model_dir = './experiments/model/{}/'.format(folder_name)
ckpt_format = model_dir + '{}.ckpt'
log_txt_path = model_dir + 'log.txt'
tensorboard_path = './experiments/tensorboard/{}'.format(folder_name)

if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

open(log_txt_path, 'w').close()

##########################################################################################################
# 데이터셋을 불러옵니다.
##########################################################################################################
train_image_data = []
train_label_data = []

for image_path in  glob.glob('./dataset/train/*'):
    gt_image = cv2.imread(image_path)
    if gt_image is None:
        print(image_path)
        continue

    if args['color']:
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2YCrCb)
    else:
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2GRAY)
    
    image = cv2.resize(gt_image, None, fx = 1/args['scale'], fy = 1/args['scale'])
    image = cv2.resize(image, None, fx = args['scale'], fy = args['scale'])

    image = image.astype(np.float32) / 255.
    gt_image = gt_image.astype(np.float32) / 255.

    h, w, c = gt_image.shape

    for y in range(0, h-args['image_size'], args['stride']):
        for x in range(0, w-args['image_size'], args['stride']):
            sub_image = image[y:y+args['image_size'], x:x+args['image_size']]
            sub_label = gt_image[y:y+args['image_size'], x:x+args['image_size']]
            
            # gray scale
            if not args['color']:
                sub_image = sub_image[..., np.newaxis]
                sub_label = sub_label[..., np.newaxis]
            
            # print(sub_image.shape, sub_image.dtype)
            train_image_data.append(sub_image)
            train_label_data.append(sub_label)

train_image_data = np.asarray(train_image_data, dtype = np.float32)
train_label_data = np.asarray(train_label_data, dtype = np.float32)

log_print('# train length : {}'.format(len(train_image_data)), log_txt_path)

##########################################################################################################
# Loss, Optimizer 등 학습에 필요한 내용들을 준비합니다.
##########################################################################################################
image_var = tf.placeholder(tf.float32, [None, args['image_size'], args['image_size'], args['color_depth']])
label_var = tf.placeholder(tf.float32, [None, args['image_size'], args['image_size'], args['color_depth']])

predictions_op = SRCNN(image_var, {
    'use_sigmoid' : True,
    
    'conv1' : dict(filters = 64, kernel_size = (9, 9), strides = 1, padding = 'same', name = 'conv1'),
    'conv2' : dict(filters = 32, kernel_size = (1, 1), strides = 1, padding = 'same', name = 'conv2'),
    'conv3' : dict(filters = args['color_depth'], kernel_size = (5, 5), strides = 1, padding = 'same', name = 'conv3'),
})

loss_op = tf.reduce_mean(tf.square(label_var - predictions_op))

if args['optimizer'] == 'SGD':
    train_op = tf.train.GradientDescentOptimizer(args['learning_rate']).minimize(loss_op)
else:
    assert False, "[!] Optimizer"

##########################################################################################################
# 디버깅을 위한 Tensorboard를 준비합니다.
##########################################################################################################
tf.summary.scalar('Loss', loss_op)
train_summary_op = tf.summary.merge_all()

##########################################################################################################
# 학습 전 GPU 메모리를 할당받습니다.
##########################################################################################################
sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver(max_to_keep = 1)

train_writer = tf.summary.FileWriter(tensorboard_path)

##########################################################################################################
# 학습 하는 부분입니다.
##########################################################################################################
train_ops = [train_op, loss_op, train_summary_op]
train_iteration = len(train_image_data) // args['batch_size']

for epoch in range(1, args['max_epoch'] + 1):
    
    loss_list = []
    train_time = time.time()

    for iter in range(train_iteration):
        _feed_dict = {
            image_var : train_image_data[iter * args['batch_size'] : (iter + 1) * args['batch_size']],
            label_var : train_label_data[iter * args['batch_size'] : (iter + 1) * args['batch_size']],
        }
        _, loss, summary = sess.run(train_ops, feed_dict = _feed_dict)
        train_writer.add_summary(summary, epoch * + iter)

        loss_list.append(loss)
    
    loss = np.mean(loss_list)
    train_time = int(time.time() - train_time)
    
    log_print('[i] epoch = {}, loss = {:.4f}, train_time = {}sec'.format(epoch, loss, train_time), log_txt_path)
    
    if epoch % args['save_epoch'] == 0:
        saver.save(sess, ckpt_format.format(epoch))

saver.save(sess, ckpt_format.format('end'))

