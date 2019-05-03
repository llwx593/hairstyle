# -*- coding: utf-8 -*-
import numpy as np
import cv2
import dlib
import matplotlib.pyplot as plt

#debug
import os

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')

#中文路径问题
def cv_imread(file_path):
    return cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),cv2.IMREAD_COLOR)

def show_68points(img_path):
    # cv2读取图像
    img = cv_imread(img_path)

    # 取灰度
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 人脸数rects
    rects = detector(img_gray, 0)
    for i in range(len(rects)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img,rects[i]).parts()])
        for idx, point in enumerate(landmarks):
            # 68点的坐标
            pos = (point[0, 0], point[0, 1])
            #print(idx,pos)

            # 利用cv2.circle给每个特征点画一个圈，共68个
            cv2.circle(img, pos, 2, color=(0, 255, 0))
            # 利用cv2.putText输出1-68
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(idx+1), pos, font, 0.35, (0, 0, 255), 1,cv2.LINE_AA)

    cv2.namedWindow("img", 2)
    cv2.imshow("img", img)
    cv2.waitKey(0)

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
    plt.axis('equal')
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

if __name__=="__main__":
    #show_68points("test_images/389.jpg")

    #show one dir
    path = 'face_classification/standard_faces/长脸/'

    img_path_list=os.listdir(path)
    counter=0
    for img_path in img_path_list:
        print(counter,len(img_path_list))
        counter+=1

        show_68points(os.path.join(path,img_path))