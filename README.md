# Tensorflow_SRCNN

## # Abstract
- Tensorflow implementation of Convolutional Neural Networks for super-resolution. The original code Matlab and Caffe from official website can be found [here](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html).

- Super-resolution is increasing resolution of image. Super-resolution exists various models. I select SRCNN. SRCNN architecture has lightweight structure. My SRCNN implemented is less than 100KB. I spend 3 hours to implement SRCNN. SRCNN use three convolution layers, so model architecture is fast and light.

- This is not officially code.

## # Hardware
- My desktop performance is Intel i7-9700K CPU, RTX 2080, and 32GB RAM.
- I think Intel i5, GTX 1050, and 8GB RAM are enough for training my project.

## # Setup
- "requirements.txt" have all library information.
- If you run my project, you may install with the following command.

```
pip install -r requirements.txt
```

## # Quick guide
- If you want to train SRCNN, you run the following command.

```
python Train_with_Thread.py
```

- If you want to test SRCNN, you run the following commands.

```
python Convert_ckpt_to_pb.py
python test/Test.py
```

- If you want to train your dataset, you change training folder "./dataset/train/".

## # Resuls
- Will be uploaded...

## # Conclusion
- If you have questions about code, please contact josanghyeokn@gmail.com

## # Reference
- https://github.com/tegg89/SRCNN-Tensorflow
- Image Super-Resolution Using Deep Convolutional Networks []