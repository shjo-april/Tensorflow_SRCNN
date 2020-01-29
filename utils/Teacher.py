# Copyright (C) 2020 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import cv2
import copy
import time
import random

import numpy as np

from threading import Thread

from utils.Utils import *
from utils.Timer import *

class Teacher(Thread):
    
    def __init__(self, option):
        Thread.__init__(self)
        
        self.train = True
        self.option = option

        self.main_queue = option['main_queue']
        
        self.image_size = option['image_size']
        self.batch_size = option['batch_size']

        self.image_paths = option['image_paths']
        self.resize_methods = [
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_AREA,
            cv2.INTER_CUBIC,
            cv2.INTER_LANCZOS4,
        ]
        
    def run(self):
        while self.train:
            while self.main_queue.full() and self.train:
                time.sleep(0.1)
                continue
            
            batch_image_data = []
            batch_label_data = []

            scale = random.randint(self.option['min_scale'], self.option['max_scale'])

            for i in range(self.batch_size * 4):
                image_path = random.choice(self.image_paths)

                gt_image = cv2.imread(image_path)
                if gt_image is None:
                    print('[!] cv2.imread({})'.format(image_path))
                    continue
                
                gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2YCrCb)

                resize_method = random.choice(self.resize_methods)
                image = cv2.resize(gt_image, None, fx = 1/scale, fy = 1/scale, interpolation = resize_method)
                image = cv2.resize(image, None, fx = scale, fy = scale, interpolation = resize_method)

                h, w, c = image.shape
                
                for i in range(self.option['crop_per_image']):
                    xmin = np.random.randint(0, w - self.image_size)
                    ymin = np.random.randint(0, h - self.image_size)
                    xmax = xmin + self.image_size
                    ymax = ymin + self.image_size

                    batch_image_data.append(image[ymin:ymax, xmin:xmax, :])
                    batch_label_data.append(gt_image[ymin:ymax, xmin:xmax, :])

                if len(batch_image_data) == self.batch_size:
                    break

            batch_image_data = np.asarray(batch_image_data, dtype = np.float32) / 255.
            batch_label_data = np.asarray(batch_label_data, dtype = np.float32) / 255.
            
            try:
                self.main_queue.put_nowait([batch_image_data, batch_label_data])
            except:
                pass
