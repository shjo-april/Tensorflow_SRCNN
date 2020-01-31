# Copyright (C) 2020 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import cv2
import glob

import numpy as np
import tensorflow as tf

class SRCNN:
    def __init__(self, model_path, gpu_usage = 0.90, strides = (14, 14)):

        def load_graph(frozen_graph_filename):
            with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            with tf.Graph().as_default() as graph:
                tf.import_graph_def(graph_def, name = 'prefix')

            return graph

        graph = load_graph(model_path)
        
        self.image_var = graph.get_tensor_by_name('prefix/images:0')
        self.predictions_op = graph.get_tensor_by_name('prefix/SRCNN/outputs:0')
        
        shape = self.image_var.shape.as_list()

        self.x_stride, self.y_stride = strides
        _, self.height, self.width, _ = shape
        
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = gpu_usage
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(graph = graph, config = config)

    def split(self, image):
        h, w = image.shape

        skip_w = w % self.width == 0
        skip_h = h % self.height == 0

        batch_images = []

        y = 0
        while (y + self.height) < h:
            x = 0
            while (x + self.width) < w:
                # print(x, y, image.shape)
                batch_images.append(image[y:y+self.height, x:x+self.width])
                x += self.x_stride

            if not skip_w:
                # print(x, y, image.shape)
                batch_images.append(image[y:y+self.height, -self.width:])

            y += self.y_stride

        if not skip_h:
            x = 0
            while (x + self.width) < w:
                # print(x, y, image.shape)
                batch_images.append(image[-self.height:, x:x+self.width])
                x += self.x_stride

            if not skip_w:
                # print(x, y, image.shape)
                batch_images.append(image[-self.height:, -self.width:])
        
        batch_images = np.asarray(batch_images, dtype = np.float32)
        batch_images = batch_images[..., np.newaxis]
        return batch_images / 255.

    def merge(self, pred_images, shape, image):
        h, w = shape

        skip_w = w % self.width == 0
        skip_h = h % self.height == 0

        index = 0
        merge_image = image.copy() # np.zeros((h, w, c), dtype = np.float32)
        merge_image = merge_image[..., np.newaxis]

        y = 0
        while (y + self.height) < h:
            x = 0
            while (x + self.width) < w:
                merge_image[y:y+self.height, x:x+self.width, :] = pred_images[index]; index += 1

                x += self.x_stride

            if not skip_w:
                merge_image[y:y+self.height, -self.width:, :] = pred_images[index]; index += 1

            y += self.y_stride

        if not skip_h:
            x = 0
            while (x + self.width) < w:
                merge_image[-self.height:, x:x+self.width, :] = pred_images[index]; index += 1
                x += self.x_stride

            if not skip_w:
                merge_image[-self.height:, -self.width:, :] = pred_images[index]; index += 1
        
        merge_image = np.asarray(merge_image, dtype = np.uint8)
        return merge_image
    
    def predict(self, image):
        batch_images = self.split(image)

        pred_images = self.sess.run(self.predictions_op, feed_dict = {self.image_var : batch_images})

        return self.merge(pred_images * 255., image.shape, image)

scale = 3
model = SRCNN('./SRCNN.pb')

def decode_image(image):
    return image

for image_path in glob.glob('./dataset/test/Set5/*'):
# for image_path in glob.glob('./dataset/train/*'):
    gt_image = cv2.imread(image_path)
    gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2GRAY)
    
    test_image = cv2.resize(gt_image, None, fx = 1/scale, fy = 1/scale)
    test_image = cv2.resize(test_image, None, fx = scale, fy = scale)

    # cv2.imshow('show', decode_image(gt_image))
    # cv2.waitKey(0)

    pred_image = model.predict(test_image)

    cv2.imshow('test', decode_image(test_image))
    cv2.imshow('SRCNN', decode_image(pred_image))
    cv2.waitKey(0)
