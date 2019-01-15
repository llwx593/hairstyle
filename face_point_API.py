import face_points as fp

import numpy as np
from skimage import io

def get_face_attributes_list(image_array):
    '''输入值域为（0，255）的图片像素矩阵，返回4个属性的列表，
    前3个属性都是数值，第4个属性是ndarray'''
    points_array = fp.get_points_array_with_image_array(image_array)
    attributes_list = fp.get_3_attributes_list(points_array)
    attributes_list.append(fp.distance_matrix(points_array))
    return attributes_list

def get_4_distance_and_norm(attributes_list_1, attributes_list_2,
                            norm_k=[20, 1, 0.2, 1]):
    '''输入两个由get_face_attributes_list（）得到的属性列表，
    返回一个表示差值的矩阵，形状（4，1），norm_k为归一化的系数'''
    distance_array = np.zeros((4, 1))
    a1 = np.array(attributes_list_1[:-1]).reshape(-1, 1)
    a2 = np.array(attributes_list_2[:-1]).reshape(-1, 1)
    distance_array[0:3, :] = a1 - a2
    distance_array[3, :] = np.array(fp.distance_with_distance_matrix(
        attributes_list_1[3], attributes_list_2[3]))
    distance_array = distance_array/(np.array(norm_k).reshape(distance_array.shape))
    return np.abs(distance_array)

def compare_face(attributes_list_1, attributes_list_2,
                 weight_list=[0.2, 0.4, 0.2, 0.2]):
    '''输入两个由get_face_attributes_list（）得到的属性列表，
    比较两张脸，得到一个表示距离的数值。w为权重，分别对应于脸颊的
    最大拐角、下巴曲率半径、宽高比、特征点完全图距离矩阵的差距。'''
    distance_array = get_4_distance_and_norm(attributes_list_1, attributes_list_2)
    distance_array = np.power(distance_array, 2)
    w = np.array(weight_list).reshape(distance_array.shape)
    return np.sum(np.multiply(distance_array, w))

path='images\\women\\*.jpg'
collections=io.ImageCollection(path)
atts = []
for i in collections:
    atts.append(get_face_attributes_list(i))
# np.random.seed(1)
# l=np.random.permutation(range(40))
atts = atts[0::2]
dist= np.zeros((4, 1))
count=0
for i in range(len(atts)):
    for j in range(i+1,len(atts)):
        count+=1
        dist = dist + get_4_distance_and_norm(atts[i], atts[j])
print(dist/count)
'''
[[ 1.        ]
 [-2.10415665]
 [ 0.31163515]
 [40.61256259]]
 '''
