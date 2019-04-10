# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 20:11:59 2019

@author: 15440
"""
from skimage import io

import os
from faceswap import output

user_image_path='test_images/56.jpg'
target_image_dir='test_images/test_man/'

target_image_path_list=os.listdir(target_image_dir)
print(target_image_path_list)

for target_image_path in target_image_path_list:
    target_image_path=os.path.join(target_image_dir,target_image_path)

    output(target_image_path,user_image_path)

    io.imshow(io.imread(user_image_path))
    io.show()
    io.imshow(io.imread(target_image_path))
    io.show()
    io.imshow(io.imread('output/output.jpg'))
    io.show()