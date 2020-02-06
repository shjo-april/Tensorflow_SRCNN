# Copyright (C) 2020 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import cv2
import glob

import numpy as np
import tensorflow as tf

class SRCNN:
    def __init__(self, model_path, gpu_usage = 0.90):
        
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

        _, self.height, self.width, _ = shape
        self.x_stride, self.y_stride = self.width, self.height
        # self.x_stride, self.y_stride = 14, 14
        
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = gpu_usage
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(graph = graph, config = config)

    def split(self, image):
        h, w, c = image.shape

        batch_images = []
        for y in range(0, h-self.height+1, self.y_stride):
            for x in range(0, w-self.width+1, self.x_stride):
                batch_images.append(image[y:y+self.height, x:x+self.width])
        
        batch_images = np.asarray(batch_images, dtype = np.float32)
        return batch_images / 255.

    def merge(self, pred_images, shape):
        pred_image = np.zeros(shape, dtype = np.float32)

        index = 0
        for y in range(0, h-self.height+1, self.y_stride):
            for x in range(0, w-self.width+1, self.x_stride):
                pred_image[y:y+self.height, x:x+self.width] = pred_images[index]
                index += 1

        return pred_image.astype(np.uint8)
    
    def predict(self, image):
        batch_images = self.split(image)

        pred_images = self.sess.run(self.predictions_op, feed_dict = {self.image_var : batch_images})
        pred_images = np.maximum(np.minimum(pred_images, 1.), 0.)

        return self.merge(pred_images * 255, image.shape)

scale = 5
model = SRCNN('./test/SRCNN.pb')

for image_path in glob.glob('./dataset/test/Set5/*'):
# for image_path in glob.glob('./dataset/train/*'):
    gt_image = cv2.imread(image_path)
    h, w, c = gt_image.shape
    
    gt_image = gt_image[:-(h % model.y_stride), :-(w % model.x_stride), :]
    h, w, c = gt_image.shape
    
    test_image = cv2.resize(gt_image, None, fx = 1/scale, fy = 1/scale)
    test_image = cv2.resize(test_image, (w, h))
    
    pred_image = model.predict(test_image)

    cv2.imshow('GT', gt_image)
    cv2.imshow('LR', test_image)
    cv2.imshow('HR', pred_image)
    cv2.waitKey(0)
