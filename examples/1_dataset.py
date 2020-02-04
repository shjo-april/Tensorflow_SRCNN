# Copyright (C) 2020 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import cv2
import glob

scale = 3
image_paths = glob.glob('./dataset/train/*')

for image_path in image_paths:
    image = cv2.imread(image_path)
    if image is None:
        print('[!] cv2.imread({})'.format(image_path))
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    
    src_image = cv2.resize(image, None, fx = 1/scale, fy = 1/scale)
    src_image = cv2.resize(src_image, None, fx = scale, fy = scale)

    print(src_image.min(), src_image.max())

    cv2.imshow('original image', image)
    cv2.imshow('resized image', src_image)
    cv2.waitKey(0)

