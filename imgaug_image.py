#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
import random
from imgaug import augmenters as iaa


gen_img = [
    iaa.PerspectiveTransform(scale=(0.002, 0.006)),
    iaa.AverageBlur(k=((3), (1, 3))),
    #iaa.AveragePooling(2),
    iaa.AddElementwise((-20, -5)),
    iaa.AdditiveGaussianNoise(scale=(0, 0.01 * 255)),
    iaa.JpegCompression(compression=(10, 30)),
    iaa.WithBrightnessChannels(iaa.Add((-50, 50)), to_colorspace=[iaa.CSPACE_Lab, iaa.CSPACE_HSV]),
    iaa.Add((10, 60), per_channel=0.5),
    iaa.Multiply((0.75, 1.25), per_channel=0.5)
]


def generate(img):
    if random.uniform(0, 1) < 0.7:
        seq = iaa.SomeOf((1), gen_img)
        image_aug = seq.augment_image(img)
        return image_aug
    elif random.uniform(0, 1) < 0.05:
        seq = iaa.SomeOf((1), [iaa.Add((10, 60), per_channel=0.5)])
        image_aug = seq.augment_image(img)
        return 255 - image_aug 
    return img


def generate_image(image):
    image = generate(image)
    return image


if __name__ == "__main__":
    path = "show/"
    img_list = os.listdir(path)
    for i, name in enumerate(img_list):
        image = cv2.imread(path+name)
        image = cv2.resize(image, (0, 0), fx=3, fy=3)
        img = generate_image(image)
        im = np.vstack((cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
        cv2.imshow('img', im)
        cv2.moveWindow('img', 300, 10)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



