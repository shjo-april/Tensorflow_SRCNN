# Copyright (C) 2020 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import cv2
import glob
import numpy as np

images = []
image_path = './dataset/test/Set5/baby_GT.bmp'

hr_image = cv2.imread(image_path)
h, w, c = hr_image.shape

images.append(hr_image)

for scale in [3, 5, 7, 9]:
    lr_image = cv2.resize(hr_image, None, fx = 1/scale, fy = 1/scale)
    lr_image = cv2.resize(lr_image, (w, h))

    text = 'Scale = {}'.format(scale)
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, 0.7, 2)

    cv2.rectangle(lr_image, (0, 0), (text_size[0], text_size[1] + 5), (0, 255, 0), cv2.FILLED)
    cv2.putText(lr_image, text, (0, text_size[1]), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 2)

    # cv2.imshow('LR image (scale = {})'.format(scale), lr_image)
    images.append(lr_image)

# cv2.imshow('HR image', hr_image)
# cv2.waitKey(0)

demo_image = np.zeros((h, w * len(images), 3), dtype = np.uint8)

for i, image in enumerate(images):
    demo_image[:, i * w : (i + 1) * w, :] = image

cv2.imwrite('./res/scale.jpg', demo_image)