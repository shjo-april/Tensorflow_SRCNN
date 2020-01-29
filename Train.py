# Copyright (C) 2020 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import os
import cv2
import glob
import argparse

import numpy as np
import tensorflow as tf

from core.SRCNN import *

from utils.Utils import *
from utils.Teacher import *

def parse_args():
    parser = argparse.ArgumentParser(description='SRCNN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--use_gpu', dest='use_gpu', help='use gpu', default='0', type=str)

    parser.add_argument('--min_scale', dest='min_scale', help='min_scale', default=3, type=int)
    parser.add_argument('--max_scale', dest='max_scale', help='max_scale', default=3, type=int)

    parser.add_argument('--image_size', dest='image_size', help='image_size', default=32, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='batch_size', default=128, type=int)

    parser.add_argument('--optimizer', dest='optimizer', help='optimizer', default='SGD', dtype=str)
    parser.add_argument('--learning_rate', dest='learning_rate', help='learning_rate', default=1e-4, type=float)
    parser.add_argument('--max_iteration', dest='max_iteration', help='max_iteration', default=150000, type=int)

    parser.add_argument('--resize_augment', dest='resize_augment', help='resize_augment', default=False, type=bool)

    return parser.parse_args()

args = vars(parse_args())

folder_name = 'SRCNN_image={}x{}_batch={}_optimizer={}_lr={}'.format(args['image_size'], args['batch_size'], args['optimizer'], args['learning_rate'])

os.environ["CUDA_VISIBLE_DEVICES"] = args['use_gpu']

def ycbcr_imread(image_path, mode = 'ycbcr'):
    image = cv2.imread(image_path)
    assert image is not None, "[!] cv2.imread({})".format(image_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    return image

# image = ycbcr_imread('./dataset/train/t1.bmp')
# cv2.imshow('show', image)
# cv2.waitKey(0)

