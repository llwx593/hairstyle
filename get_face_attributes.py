# -*- coding: utf-8 -*-

import dlib
import numpy as np
import math
from skimage import io


class NoFaceException(Exception):
    pass
class TooManyFaces(Exception):
    pass

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')

'''2 获取脸型图片特征点'''
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

# 保持长宽比例不变，把外切矩形的面积转换成1
def change_size(points_array):
    x_distance = (points_array[:, 0].max() - points_array[:, 0].min())
    y_distance = (points_array[:, 1].max() - points_array[:, 1].min())
    k = math.sqrt(x_distance*y_distance)
    return points_array/k

# 输入特征点矩阵，（68，2），输出把中心挪到原点后的特征点矩阵
def move_to_center(points_array):
    cen = np.sum(points_array,axis=0)/ points_array.shape[0]
    return points_array - cen

def get_normalized_points_array(image_array):
    return change_size(move_to_center(get_points_array(image_array)))

def get_face_edge_points(points_array):
    return points_array[0:17, :]

'''3 计算脸部属性'''
def linear_fit(x,y):
    x_aver=np.mean(x)
    y_aver=np.mean(y)
    x2_aver=np.mean(np.multiply(x,x))
    xy_aver=np.mean(np.multiply(x,y))
    if x2_aver-x_aver**2<0.0001:
        return (xy_aver-x_aver*y_aver)/(x2_aver-x_aver**2+0.0001)
    return (xy_aver-x_aver*y_aver)/(x2_aver-x_aver**2)

def gradient(point1, point2):
    if abs(point2[0, 0]-point1[0, 0]) < 0.0001:
        return (point2[0, 1]-point1[0, 1])/(point2[0, 0]-point1[0, 0] + 0.0001)
    return (point2[0, 1]-point1[0, 1])/(point2[0, 0]-point1[0, 0])

#脸部拐点角 反映国字脸
def cheek_degree(points_array):
    k1=linear_fit(points_array[:5,0],points_array[:5,1])
    k2=linear_fit(points_array[5:9,0],points_array[5:9,1])

    k3=linear_fit(points_array[12:17,0],points_array[12:17,1])
    k4=linear_fit(points_array[10:14,0],points_array[10:14,1])

    theta1=180-(math.atan(k1)-math.atan(k2))/math.pi*180
    theta2=180-(math.atan(k3)-math.atan(k4))/math.pi*180
    return (theta1+theta2)/2

# 下巴曲率半径 反映下巴形状 尖下巴 圆下巴
def jaw_curvature(points_array):
    #下巴采用 6-12的特征点
    jaw = points_array[5:11, :]
    x = jaw[:, 0]
    y = jaw[:, 1]
    a = np.sum(np.multiply(np.power(x, 2), y))/(x.shape[0])
    b = np.sum(np.power(x, 2))/(x.shape[0])
    c = np.sum(y)/(y.shape[0])
    d = np.sum(np.power(x, 4))/(x.shape[0])
    R = -2*((d - b**2)/(a - b*c))
    return R
# 长宽比 长不是额头到下巴 反映脸的长短 长脸或其它
def heigth_divide_width(points_array):
    '''由于数据点不含额头，这里的高是第一个特征点到下巴距离'''
    height =-(points_array[0,1]+points_array[16,1])/2+\
            (points_array[7,1]+points_array[8,1]+points_array[9,1])/3
    width=0
    for i in range(5):
        width+=points_array[16-i,0]-points_array[i,0]
    width/=5
    return height/width
#反映脸颊倾斜程度 瓜子脸
def width_divide_width(points_array):
    width1=( (points_array[16,0]-points_array[0,0])+\
            (points_array[15,0]-points_array[1,0]) )/2

    width2=(points_array[12,0]-points_array[4,0])
    return width1/width2
def get_attributes_list(image_array):
    f=get_normalized_points_array(image_array)
    f=get_face_edge_points(f);
    atrributes = []
    atrributes.append(cheek_degree(f))
    atrributes.append(jaw_curvature(f))
    atrributes.append(heigth_divide_width(f))
    atrributes.append(width_divide_width(f))
#    print(atrributes)
    return atrributes

if __name__=='__main__':
    get_attributes_list(io.imread('test_images/22.jpg'))