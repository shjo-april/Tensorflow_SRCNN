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
from utils.Teacher import *

def parse_args():
    parser = argparse.ArgumentParser(description='SRCNN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--use_gpu', dest='use_gpu', help='use gpu', default='0', type=str)
    
    parser.add_argument('--min_scale', dest='min_scale', help='min_scale', default=2, type=int)
    parser.add_argument('--max_scale', dest='max_scale', help='max_scale', default=5, type=int)
    parser.add_argument('--crop_per_image', dest='crop_per_image', help='crop_per_image', default=16, type=int)
    
    parser.add_argument('--image_size', dest='image_size', help='image_size', default=33, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='batch_size', default=128, type=int)
    
    parser.add_argument('--optimizer', dest='optimizer', help='optimizer', default='Momentum', type=str)
    parser.add_argument('--learning_rate', dest='learning_rate', help='learning_rate', default=1e-4, type=float)
    
    parser.add_argument('--num_threads', dest='num_threads', help='num_threads', default=6, type=int)
    
    parser.add_argument('--log_iteration', dest='log_iteration', help='log_iteration', default=100, type=int)
    parser.add_argument('--save_iteration', dest='save_iteration', help='save_iteration', default=10000, type=int)
    parser.add_argument('--max_iteration', dest='max_iteration', help='max_iteration', default=2550000, type=int)

    return parser.parse_args()

##########################################################################################################
# 모든 파라미터를 준비하고, 중요한 정보는 저장합니다.
##########################################################################################################
args = vars(parse_args())

os.environ["CUDA_VISIBLE_DEVICES"] = args['use_gpu']

folder_name = 'SRCNN_image={}x{}_batch={}_optimizer={}_lr={}'.format(args['image_size'], args['image_size'], args['batch_size'], args['optimizer'], args['learning_rate'])

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
image_paths = glob.glob('./dataset/train/*')

log_print('# train length : {}'.format(len(image_paths)), log_txt_path)

##########################################################################################################
# Loss, Optimizer 등 학습에 필요한 내용들을 준비합니다.
##########################################################################################################
image_var = tf.placeholder(tf.float32, [None, args['image_size'], args['image_size'], 3])
label_var = tf.placeholder(tf.float32, [None, args['image_size'], args['image_size'], 3])

predictions_op = SRCNN(image_var, {
    'use_sigmoid' : False,
    
    'conv1' : dict(filters = 64, kernel_size = (9, 9), strides = 1, padding = 'same', kernel_initializer = tf.contrib.layers.xavier_initializer(), name = 'conv1'),
    'conv2' : dict(filters = 32, kernel_size = (1, 1), strides = 1, padding = 'same', kernel_initializer = tf.contrib.layers.xavier_initializer(), name = 'conv2'),
    'conv3' : dict(filters = 3, kernel_size = (5, 5), strides = 1, padding = 'same', kernel_initializer = tf.contrib.layers.xavier_initializer(), name = 'conv3'),
})

loss_op = tf.reduce_mean(tf.square(label_var - predictions_op))

if args['optimizer'] == 'SGD':
    train_op = tf.train.GradientDescentOptimizer(args['learning_rate']).minimize(loss_op)
elif args['optimizer'] == 'Momentum':
    train_op = tf.train.MomentumOptimizer(args['learning_rate'], momentum = 0.9, use_nesterov = True).minimize(loss_op)
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
# 데이터 전처리를 빠르게 도와주는 스레드를 생성합니다.
##########################################################################################################
main_queue = Queue(100 * args['num_threads'])

thread_option = {
    'main_queue' : main_queue,
    'image_size' : args['image_size'],
    'batch_size' : args['batch_size'],

    'image_paths' : image_paths,

    'min_scale' : args['min_scale'],
    'max_scale' : args['max_scale'],

    'crop_per_image' : args['crop_per_image'],
}

train_threads = []
for i in range(args['num_threads']):
    th = Teacher(thread_option)
    th.start()

    train_threads.append(th)

##########################################################################################################
# 학습 하는 부분입니다.
##########################################################################################################
train_ops = [train_op, loss_op, train_summary_op]

loss_list = []
train_time = time.time()

for iter in range(1, args['max_iteration'] + 1):
    batch_image_data, batch_label_data = main_queue.get()

    _feed_dict = {
        image_var : batch_image_data,
        label_var : batch_label_data,
    }
    _, loss, summary = sess.run(train_ops, feed_dict = _feed_dict)

    loss_list.append(loss)

    if iter % args['log_iteration'] == 0:
        loss = np.mean(loss_list)
        train_time = int(time.time() - train_time)
        
        log_print('[i] iter = {}, loss = {:.4f}, train_time = {}sec'.format(iter, loss, train_time), log_txt_path)
        train_writer.add_summary(summary, iter)
        
        loss_list = []
        train_time = time.time()

    if iter % args['save_iteration'] == 0:
        saver.save(sess, ckpt_format.format(iter))

for th in train_threads:
    th.train = False
    th.join()

saver.save(sess, ckpt_format.format('end'))