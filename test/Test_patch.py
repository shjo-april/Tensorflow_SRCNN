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
        
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = gpu_usage
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(graph = graph, config = config)

    def split(self, image):
        h, w, c = image.shape

        batch_images = []
        for y in range(0, h-self.height, self.y_stride):
            for x in range(0, w-self.width, self.x_stride):
                batch_images.append(image[y:y+self.height, x:x+self.width])
        
        batch_images = np.asarray(batch_images, dtype = np.float32)
        return batch_images / 255.
    
    def merge(self, pred_images, shape):
        h, w, c = shape

        skip_w = w % self.width == 0
        skip_h = h % self.height == 0

        index = 0
        merge_image = np.zeros((h, w, c), dtype = np.float32)

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

        batch_images, pred_images = batch_images * 255, pred_images * 255
        return batch_images.astype(np.uint8), pred_images.astype(np.uint8)

scale = 3
model = SRCNN('./SRCNN.pb')

# for image_path in glob.glob('./dataset/test/Set5/*'):
for image_path in glob.glob('./dataset/train/*'):
    gt_image = cv2.imread(image_path)
    gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2YCrCb)
    
    h, w, c = gt_image.shape

    test_image = cv2.resize(gt_image, None, fx = 1/scale, fy = 1/scale)
    test_image = cv2.resize(test_image, (w, h))

    bg_image = np.zeros((h, w, c), dtype = np.uint8)
    bg_pred_image = np.zeros((h, w, c), dtype = np.uint8)

    images, pred_images = model.predict(test_image)
    
    i = 0
    for y in range(0, h-model.height, model.y_stride):
        for x in range(0, w-model.width, model.x_stride):
            bg_image[y:y+model.height, x:x+model.width] = images[i]
            bg_pred_image[y:y+model.height, x:x+model.width] = pred_images[i]
            i += 1
            
            cv2.imshow('Input', decode_image(bg_image))
            cv2.imshow('SRCNN', decode_image(bg_pred_image))
            cv2.waitKey(0)
