# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 17:55:17 2019

@author: 15440
"""

from skimage import io
import tensorflow as tf

path='./images/man/*.jpg'
collection=io.ImageCollection(path)

print(collection[0])
#num_picture=len(collection)
#for i in range(num_picture):
#    io.imshow(collection[i])
#    io.show()