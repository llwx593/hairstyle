# -*- coding: utf-8 -*-

import dlib
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import math
class NoFaceException(Exception):
    def __init__(self):
        pass

predictor_path = r"model\shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
def try_model0(image_array):
    win = dlib.image_window()
    img = image_array
    
    win.clear_overlay()
    win.set_image(img)
    dets = detector(img, 1)
    
    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        shape = predictor(img, d)
        landmark = np.matrix([[p.x, p.y] for p in shape.parts()])
        # print("face_landmark:")
        
        # print(landmark.shape)
        #print (landmark)  # 打印关键点矩阵
        win.add_overlay(shape)  #绘制特征点
    dlib.hit_enter_to_continue()
    
def try_model(faces_path = r"images\3.jpg"):
    img = io.imread(faces_path)
    
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        shape = predictor(img, d)
        landmark = np.matrix([[p.x, p.y] for p in shape.parts()])
        
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
#脸颊部斜角
def cheek_degree(face_point):
    face = face_point
    grads = []
    for i in range(2, 5):
        grads.append(abs(np.arctan(gradient(face[i], face[i+1])) - np.arctan(gradient(face[i+1], face[i+2]))))
    # print(grads)
    return (math.pi-(max(grads)))/math.pi*180
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

def width_divide_heigth(points_array, num = 3):
    width = []
    for i in range(num):
        width.append(points_array[16-i, 0] - points_array[i, 0])
    #print(width)
    wid = sum(width)/num
    # wid = (points_array[:, 0].max() - points_array[:, 0].min())
    height = (points_array[:, 1].max() - points_array[:, 1].min())
    return wid/height

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

def get_points_array_with_image_array(image_array):
    # 输入一个值域为(0,255)的图片像素矩阵,返回平移缩放后的特征点矩阵
    dets = detector(image_array, 1)
    if(len(dets)==0):
        raise NoFaceException()
    for k, d in enumerate(dets):
        shape = predictor(image_array, d)
        landmark = np.matrix([[p.x, p.y] for p in shape.parts()])
        return change_size(move_to_center(landmark))#只返回第一张脸


def get_points_array(image_path):
    
    img = io.imread(image_path)     # img 是一个图片像素矩阵，值域(0,255)

    dets = detector(img, 1)

    for k, d in enumerate(dets):
        shape = predictor(img, d)
        landmark = np.matrix([[p.x, p.y] for p in shape.parts()])
        return landmark#只返回第一张脸
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
def distance1(array0,array1):
    p1=np.max(array0,axis=0)
    p2=np.min(array0,axis=0)
    p1=(p1-p2)
    S=p1[0]*p1[1]
    array0=array0/np.sqrt(S)
    
    
    m=array0.shape[0]
    
    xy0_2=np.sum(np.power(array0,2),axis=0,keepdims=True)/m
    xy0=np.sum(array0,axis=0,keepdims=True)/m
    
    xy1_2=np.sum(np.power(array1,2),axis=0,keepdims=True)/m
    xy1=np.sum(array1,axis=0,keepdims=True)/m
    
    xy01=np.sum(array0*array1,axis=0,keepdims=True)/m
    
    temp=(xy0_2*xy1_2-np.power(xy01,2)-\
          np.power(xy0*xy1_2-xy1*xy01,2)/(xy1_2-np.power(xy1,2)))/xy1_2
    d=np.sum(temp)
    return np.sqrt(d)
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

def distance_with_distance_matrix(distance_matrix_1, distance_matrix_2):
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
    return distance     # n*n 维的矩阵，就是完全图的距离矩阵
#居中归一
def get_changed_points_array(face_path):
    return change_size(move_to_center(get_points_array(face_path)))

def get_face_edge_points(face_points, include_nose = False):
    if(include_nose):
        f = np.zeros((26, 2))
        f[0:17, :] = face_points[0:17, :]
        f[17:26, :] = face_points[27:36, :]
        return f
    else:
        return face_points[0:17, :]
def face_distance(face_path_1, face_path_2):
    face_1 = get_changed_points_array(face_path_1)
    face_2 = get_changed_points_array(face_path_2)
    f1 = np.zeros((26, 2))
    f2 = np.zeros((26, 2))
    f1[0:17, :] = face_1[0:17, :]
    f1[17:26, :] = face_1[27:36, :]
    f2[0:17, :] = face_2[0:17, :]
    f2[17:26, :] = face_2[27:36, :]

    return distance(f1, f2)
    #show_two_points(f1, f2)

def get_3_attributes_list(points_array):
    f = get_face_edge_points(points_array, include_nose=False)
    atrributes = []
    # cheek_degree
    atrributes.append(cheek_degree(f))
    atrributes.append(jaw_curvature(f))
    atrributes.append(width_divide_heigth(points_array, num=3))
    # print(atrributes)
    return atrributes


def evaluate(points_array1,points_array2,weight_list):
    f1=get_face_edge_points(points_array1,include_nose = False)
    f2=get_face_edge_points(points_array2,include_nose = False)
    
    atrributes1=[]
    atrributes2=[]
    #cheek_degree
    atrributes1.append(cheek_degree(f1))
    atrributes2.append(cheek_degree(f2))
    
    atrributes1.append(jaw_curvature(f1))
    atrributes2.append(jaw_curvature(f2))
    
    atrributes1.append(width_divide_heigth(points_array1, num = 3))
    atrributes2.append(width_divide_heigth(points_array2, num = 3))

    a1=np.array(atrributes1).reshape(-1,1)
    a2=np.array(atrributes2).reshape(-1,1)
    
    a1=a1/np.array([[20*0.15],[0.276],[0.2*0.47]])
    a2=a2/np.array([[20*0.15],[0.276],[0.2*0.47]])
    w=np.array(weight_list).reshape(-1,1)
    

    return np.sum(np.multiply(np.power(a1-a2,2),w))


def evaluate2(points_array1,points_array2,weight_list):
    f1=get_face_edge_points(points_array1,include_nose = False)
    f2=get_face_edge_points(points_array2,include_nose = False)
    atrributes1=[]
    atrributes2=[]
    #cheek_degree
    atrributes1.append(cheek_degree(f1))
    atrributes2.append(cheek_degree(f2))
    
    atrributes1.append(jaw_curvature(f1))
    atrributes2.append(jaw_curvature(f2))
    
    atrributes1.append(width_divide_heigth(points_array1, num = 3))
    atrributes2.append(width_divide_heigth(points_array2, num = 3))
    # print(atrributes1)
    a1=np.array(atrributes1).reshape(-1,1)
    a2=np.array(atrributes2).reshape(-1,1)
    
    a1=a1/np.array([[20],[1],[0.2]])
    a2=a2/np.array([[20],[1],[0.2]])
    print(a1 - a2)
    print(distance(points_array1, points_array2))
    w=np.array(weight_list).reshape(-1,1)
    # 先弄一个平方之前的[a1-a2, distance], 然后确定一个[4,1]的系数，再平方放大
    delta_square=np.abs(a1-a2)**2
    dist_square=distance(f1,f2)**2
    delta_square=np.append(delta_square,[[dist_square]],axis=0) # shape = [4, 1] , numpy.ndarray
    
    return np.sum(np.multiply(delta_square,w)),delta_square    # 到时候要修改
#
# #def recommand():
#
# user_face_path='images\\man\\u=172972933,2291792799&fm=26&gp=0.jpg'
# face_points0=get_changed_points_array(user_face_path)
#
# L=[]
# #脸颊角度 下巴曲率半径 脸部长宽比 distance
# WEIGHT=[0.2,0.4,0.2,0.2]
# #WEIGHT=[0.3,0.4,0.3]
#
# path='images\\man\\*.jpg'
# collections=io.ImageCollection(path)
# print(collections)
#
# draw_1 = []
# draw_2 = []
# delta_square_all=[]
# for i in ['images\\man\\u=172972933,2291792799&fm=26&gp=0.jpg', 'images\\man\\u=283846838,3017887413&fm=26&gp=0.jpg', 'images\\man\\u=1301088983,636709627&fm=26&gp=0.jpg', 'images\\man\\u=1627006444,3336284082&fm=26&gp=0.jpg', 'images\\man\\u=2506157847,2674135355&fm=26&gp=0.jpg', 'images\\man\\u=2623628823,3324713832&fm=26&gp=0.jpg', 'images\\man\\u=2932027764,1369049847&fm=26&gp=0.jpg', 'images\\man\\u=3123600464,1058750482&fm=26&gp=0.jpg', 'images\\man\\u=3568392546,729156733&fm=11&gp=0.jpg', 'images\\man\\u=3920783153,427755879&fm=26&gp=0.jpg', 'images\\man\\u=3991619722,3767876164&fm=26&gp=0.jpg', 'images\\man\\u=4200512207,3956956204&fm=26&gp=0.jpg']:
#     print("===================",i,"======================")
#     face_path=i
#
#     face_points=get_changed_points_array(face_path)
#
#     dist=face_distance(user_face_path,i)
#     draw_1.append(dist*25)
#
#     e,delta_square=evaluate2(face_points0,face_points,WEIGHT)
#     delta_square_all.append(delta_square)
#
#     L.append([e, i])
#     draw_2.append(e)
#
# #
#     #show_two_points(get_face_edge_points(face_points0),get_face_edge_points(face_points))
#     #face_distance(user_face_path, face_path)
#     #try_model(face_path)
#
# '''打印最后排名'''
# #L.sort()
# #i=0
# #for e in L:
# #    print("====================", i, "======================")
# #    print(e[0])
# #    io.imshow(e[1])
# #    io.show()
# #    i+=1
#
# #plt.plot(range(len(draw_1)), draw_1, "r")
# #plt.plot(range(len(draw_2)), draw_2, "b")
# #plt.show()
#
# delta_square_all_array=np.array(delta_square_all).reshape(4,-1)
# print(delta_square_all_array.shape)
# s=np.sum(delta_square_all_array,axis=1,keepdims=True)/delta_square_all_array.shape[1]
# print(s)
#
