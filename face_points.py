# -*- coding: utf-8 -*-

import dlib
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import math

predictor_path = r"model\shape_predictor_68_face_landmarks.dat"
faces_path = r"images\l1.jpg"

# '''加载人脸检测器、加载官方提供的模型构建特征提取器'''
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(predictor_path)
#
# win = dlib.image_window()
# img = io.imread(faces_path)
#
# win.clear_overlay()
# win.set_image(img)
#
# dets = detector(img, 1)
# print("Number of faces detected: {}".format(len(dets)))
# print("aaaaaa")
# for k, d in enumerate(dets):
#     print("bbbbb")
#     shape = predictor(img, d)
#     landmark = np.matrix([[p.x, p.y] for p in shape.parts()])
#     print("face_landmark:")
#
#     print(landmark.shape)
#     print (landmark)  # 打印关键点矩阵
#     win.add_overlay(shape)  #绘制特征点
# x = []
# y = []
# for i in range(27,37):
#     y.append(landmark[i, 1])
#     x.append(landmark[i, 0])
# plt.plot(x, y, "b*")
# plt.show()
# dlib.hit_enter_to_continue()

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
    plt.plot(x, y, "b*")
    # plt.plot(x_cen, y_cen, "r")

    x = []
    y = []
    for i in range(points_array_2.shape[0]):
        y.append(points_array_2[i, 1])
        x.append(points_array_2[i, 0])
    cen = points_array_2.sum(axis=0) / points_array_2.shape[0]
    # x_cen = cen[0, 0]
    # y_cen = cen[0, 1]
    plt.plot(x, y, "g*")
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
    # return np.sqrt(np.sum(np.power(points_array_1-points_array_2, 2))/points_array_1.shape[0])

    derta = (points_array_1-points_array_2).reshape(-1, 1)
    return np.sqrt(((np.sum(np.power(derta, 2)))/derta.shape[0]) - np.power((np.sum(derta)/derta.shape[0]), 2))

    # A = points_array_1.reshape(-1, 1)
    # B = points_array_2.reshape(-1, 1)
    # num = float(np.dot(A.T, B))  # 若为行向量则 A * B.T
    # denom = np.linalg.norm(A) * np.linalg.norm(B)
    # cos = num / denom  # 余弦值
    # return 1 - cos

def face_distance(face_path_1, face_path_2):
    face_1 = change_size(move_to_center(get_points_array(face_path_1)))
    face_2 = change_size(move_to_center(get_points_array(face_path_2)))
    # f1 = np.zeros((26, 2))
    # f2 = np.zeros((26, 2))
    f1 = face_1[0:17, :]
    # f1[17:26, :] = face_1[27:36, :]
    f2 = face_2[0:17, :]
    # f2[17:26, :] = face_2[27:36, :]

    print(distance(f1, f2))
    show_two_points(f1, f2)

    return distance(f1, f2)



for i in [12]:
    print("l1", i, face_distance(r"images\5.jpg", "images\\" + str(i)+ ".jpg" ))