# -*- coding: utf-8 -*-

import dlib
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import math

predictor_path = r"model\shape_predictor_68_face_landmarks.dat"


def try_model(faces_path = r"images\3.jpg"):
    '''加载人脸检测器、加载官方提供的模型构建特征提取器'''
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    # win = dlib.image_window()
    img = io.imread(faces_path)
    #
    # win.clear_overlay()
    # win.set_image(img)

    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        shape = predictor(img, d)
        landmark = np.matrix([[p.x, p.y] for p in shape.parts()])
        # print("face_landmark:")
        #
        # print(landmark.shape)
        # print (landmark)  # 打印关键点矩阵
        # win.add_overlay(shape)  #绘制特征点
    x = []
    y = []
    for i in range(68):
        y.append(landmark[i, 1])
        x.append(landmark[i, 0])
        if i>=2 and i<=6:
            plt.plot(landmark[i, 0], landmark[i, 1], "r*")
        else:
            plt.plot(landmark[i, 0], landmark[i, 1], "b*")
    plt.show()
    # dlib.hit_enter_to_continue()

def gradient(point1, point2, E = 0.001):
    if abs(point2[0, 0]-point1[0, 0]) < 0.0005:
        return (point2[0, 1]-point1[0, 1])/(point2[0, 0]-point1[0, 0] + E)
    return (point2[0, 1]-point1[0, 1])/(point2[0, 0]-point1[0, 0])

def show_gradient(face_path):
    face = get_changed_points_array(face_path)
    grads = []
    for i in range(2, 5):
        grads.append(abs(np.arctan(gradient(face[i], face[i+1])) - np.arctan(gradient(face[i+1], face[i+2]))))
    # print(grads)
    print(max(grads))

# 下巴曲率半径
def jaw_curvature(points_array):
    jaw = points_array[4:13, :]
    x = jaw[:, 0]
    y = jaw[:, 1]
    a = np.sum(np.multiply(np.power(x, 2), y))/(x.shape[0])
    b = np.sum(np.power(x, 2))/(x.shape[0])
    c = np.sum(y)/(y.shape[0])
    d = np.sum(np.power(x, 4))/(x.shape[0])
    R = -2*((d - b**2)/(a - b*c))
    return R

# def width_divide_heigth(points_array, num = 3):
#     width = []
#     for i in range(num):
#         width.append(points_array[i: 0] - points_array[16-i: 0])
#     print(width)
#     wid = sum(width)/num
#     # wid = (points_array[:, 0].max() - points_array[:, 0].min())
#     height = (points_array[:, 1].max() - points_array[:, 1].min())
#     return wid/height

def show_points(points_array):
    x = []
    y = []
    for i in range(points_array.shape[0]):
        y.append(points_array[i, 1])
        x.append(points_array[i, 0])
    cen = points_array.sum(axis=0) / points_array.shape[0]
    x_cen = cen[0, 0]
    y_cen = cen[0, 1]
    plt.plot(x, y, "b*")
    plt.plot(x_cen, y_cen, "r*")
    plt.show()

def show_two_points(points_array_1, points_array_2):
    x = []
    y = []
    for i in range(points_array_1.shape[0]):
        y.append(points_array_1[i, 1])
        x.append(points_array_1[i, 0])
    cen = points_array_1.sum(axis=0) / points_array_1.shape[0]
    # x_cen = cen[0, 0]
    # y_cen = cen[0, 1]
    plt.plot(x, y, "b")
    # plt.plot(x_cen, y_cen, "r")

    x = []
    y = []
    for i in range(points_array_2.shape[0]):
        y.append(points_array_2[i, 1])
        x.append(points_array_2[i, 0])
    cen = points_array_2.sum(axis=0) / points_array_2.shape[0]
    # x_cen = cen[0, 0]
    # y_cen = cen[0, 1]
    plt.plot(x, y, "g")
    # plt.plot(x_cen, y_cen, "b")

    plt.show()

def get_points_array(image_path, predictor_path = predictor_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    img = io.imread(image_path)


    dets = detector(img, 1)

    for k, d in enumerate(dets):
        shape = predictor(img, d)
        landmark = np.matrix([[p.x, p.y] for p in shape.parts()])
        return landmark
    return None

# 保持长宽比例不变，把外切矩形的面积转换成1
def change_size(points_array):
    x_distance = (points_array[:, 0].max() - points_array[:, 0].min())
    y_distance = (points_array[:, 1].max() - points_array[:, 1].min())
    k = math.sqrt((1/(x_distance*y_distance)))
    return points_array*k


# 输入特征点矩阵，（68，2），输出把中心挪到原点后的特征点矩阵
def move_to_center(points_array):
    cen = points_array.sum(axis = 0) / points_array.shape[0]
    return points_array - cen

def distance(points_array_1, points_array_2):
    # 欧式距离
    # return np.sqrt(np.sum(np.power(points_array_1-points_array_2, 2))/points_array_1.shape[0])

    # 平移后欧斯距离最小值
    # derta = (points_array_1-points_array_2).reshape(-1, 1)
    # return np.sqrt(((np.sum(np.power(derta, 2)))/derta.shape[0]) - np.power((np.sum(derta)/derta.shape[0]), 2))

    # cos距离
    # A = points_array_1.reshape(-1, 1)
    # B = points_array_2.reshape(-1, 1)
    # num = float(np.dot(A.T, B))  # 若为行向量则 A * B.T
    # denom = np.linalg.norm(A) * np.linalg.norm(B)
    # cos = num / denom  # 余弦值
    # return 1 - cos

    distance_matrix_1 = distance_matrix(points_array_1)
    distance_matrix_2 = distance_matrix(points_array_2)
    distance = np.sum(np.power(distance_matrix_1 - distance_matrix_2, 4))
    return np.sqrt(np.sqrt(distance))

def distance_matrix(points_array):
    ones = np.ones((1, points_array.shape[0]))
    x = (points_array[:, 0]).reshape(-1, 1)
    y = (points_array[:, 1]).reshape(-1, 1)
    x = np.dot(x, ones)
    y = np.dot(y, ones)
    x_distance = np.power(x-(x.T), 2)
    y_distance = np.power(y-(y.T), 2)
    distance = np.sqrt(x_distance + y_distance)
    return distance

def get_changed_points_array(face_path):
    return change_size(move_to_center(get_points_array(face_path)))

def face_distance(face_path_1, face_path_2):
    face_1 = change_size(move_to_center(get_points_array(face_path_1)))
    face_2 = change_size(move_to_center(get_points_array(face_path_2)))
    f1 = np.zeros((26, 2))
    f2 = np.zeros((26, 2))
    f1[0:17, :] = face_1[0:17, :]
    f1[17:26, :] = face_1[27:36, :]
    f2[0:17, :] = face_2[0:17, :]
    f2[17:26, :] = face_2[27:36, :]

    print(distance(f1, f2))
    show_two_points(f1, f2)

    return distance(f1, f2)



for i in range(22):
    print("===================",i,"======================")
    show_gradient("images\\" + str(i)+ ".jpg")
    print(jaw_curvature(get_changed_points_array("images\\" + str(i)+ ".jpg")))
    print(width_divide_heigth(get_changed_points_array("images\\" + str(i)+ ".jpg")))
    # try_model("images\\" + str(i)+ ".jpg")
#     print(i, face_distance(r"images\5.jpg", "images\\" + str(i)+ ".jpg" ))