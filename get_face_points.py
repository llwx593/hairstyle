# -*- coding: utf-8 -*-
import dlib
import numpy as np
from skimage import io

from face_swap import transformation_from_points

class NoFaceException(Exception):
    pass
class TooManyFaces(Exception):
    pass

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')

standard_face_path='database/standard_face.jpg'

def get_points_array(image_array):
    # image_array 是一个图片像素矩阵，值域(0,255)
    faces = detector(image_array, 1)
    if len(faces)==0:
        raise NoFaceException
    if len(faces)>1:
        raise TooManyFaces

    shape = predictor(image_array, faces[0])
    landmark = np.matrix([[p.x, p.y] for p in shape.parts()])
    return landmark

def get_face_edge_points(points_array):
    return points_array[0:17, :]

def get_normalized_points_array(image_array):
    points_array= get_face_edge_points(get_points_array(image_array))
    cen = np.mean(points_array,axis=0)
    var= np.mean(np.multiply(points_array-cen,points_array-cen))
    return (points_array-cen)/np.sqrt(var)

standard_face_points_array=get_normalized_points_array(
                            io.imread(standard_face_path))
def get_rotated_points_array(image_array):
    p=get_normalized_points_array(image_array)
    M=transformation_from_points(standard_face_points_array,p)
    p=np.matmul(p,M[:2,:2])
    return p


