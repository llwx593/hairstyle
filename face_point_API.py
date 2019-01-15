import face_points as fp
from face_points import NoFaceException

import numpy as np
from skimage import io
import json

def get_face_attributes_list(image_array,count):
    '''输入值域为（0，255）的图片像素矩阵，返回4个属性的列表，
    前3个属性都是数值，第4个属性是ndarray'''
    points_array = fp.get_points_array_with_image_array(image_array,count)
    attributes_list = fp.get_3_attributes_list(points_array)
    attributes_list.append(fp.distance_matrix(points_array))
    return attributes_list

def get_4_distance_and_norm(attributes_list_1, attributes_list_2):
    '''输入两个由get_face_attributes_list（）得到的属性列表，
    最后返回一个表示差值的矩阵，形状（4，1），中途将数据处理成标准正态分布'''
    distance_array = np.zeros((4, 1))
    a1 = np.array(attributes_list_1[:-1]).reshape(-1, 1)
    a2 = np.array(attributes_list_2[:-1]).reshape(-1, 1)
    distance_array[0:3, :] = a1 - a2
    distance_array[3, :] = np.array(fp.distance_with_distance_matrix(
        attributes_list_1[3], attributes_list_2[3]))
    '''得到均值mean和标准差variance_sqrt后经过下面一步处理再输出'''
    #distance_array=(distance_array-mean)/variance_sqrt
    return distance_array

def compare_face(attributes_list_1, attributes_list_2,
                 weight_list=[0.2, 0.3, 0.3, 0.2]):
    '''输入两个由get_face_attributes_list（）得到的属性列表，
    比较两张脸，得到一个表示距离的数值。w为权重，分别对应于脸颊的
    最大拐角、下巴曲率半径、宽高比、特征点完全图距离矩阵的差距。'''
    distance_array = get_4_distance_and_norm(attributes_list_1, attributes_list_2)
    distance_array = np.power(distance_array, 2)
    w = np.array(weight_list).reshape(distance_array.shape)
    return np.sum(np.multiply(distance_array, w))

'''匹配脸型,排序输出'''
# user_face_image_path='images\\20174164.jpg'
# path='student_images\\*.jpg'
# collections=io.ImageCollection(path)
# collections_user=io.ImageCollection(user_face_image_path)

# user_face_image=io.imread(user_face_image_path)

# result = []
# count=0
# for image in collections[100:200]:
#     if(count%10==0):
#         print(count)
#     count+=1
#     try:
#         attribute_list= get_face_attributes_list(image)
#         result.append( [ compare_face(user_face_image, attribute_list) , image, 
#             get_4_distance_and_norm(get_face_attributes_list(user_face_image),
#                                     get_face_attributes_list(image) ) ] )
#     except:
#         pass
# result.sort()
# i=0
# for e in result:
#     print('=================',i,'===================')
#     i+=1
#     print(e[0], e[2].reshape(1,-1))

#     io.imshow(e[1])
#     io.show()
'''获得调节4个属性向量的均值与方差'''
# #获得图片
# path='./test_images/*.jpg'
# coll=io.ImageCollection(path)
# #获得所有图片的属性列表
# attributes=[]
# count=0
# for image in coll:
#     try:
#         attributes.append(get_face_attributes_list(image,count))
#         if(count%10==0):
#                 print(count)
#     except NoFaceException as e:
#         print("未探测到脸,图片编号："+str(e.count+20174000))
#     finally:
#         count+=1
# #获得test_images中每两张图片的差值向量，存在Result中
# Result=[]
# count=0
# for i in range(len(attributes)):
#     for j in range(i+1,len(attributes)):
#         if(count%1000==0):
#             print(count)
#         count+=1
#         dist_4=get_4_distance_and_norm(attributes[i], attributes[j])
#         Result.append(dist_4)
# #由result计算每一个属性分布的均值与方差
# Result=np.array(Result).reshape(-1,4)
# mean=np.sum(Result,axis=0,keepdims=True)/Result.shape[0]
# variance=np.sum((Result-mean)**2,axis=0,keepdims=True)/Result.shape[0]
# variance_sqrt=np.sqrt(variance)
# print(mean)
# print(variance_sqrt)
# '''
# 测试了970多张图片得到的结果，但发现大量图片不清晰，故暂时不用，看能不能更换图片重算
# [[-0.07086281 -0.012985   -0.00115657  0.33695413]]
# [[4.2680468  0.26716674 0.0859714  0.10235114]]
# '''
'''筛选出像素高的图片'''
# path='./test_images/*.jpg'
# coll=io.ImageCollection(path)
# i=0
# name_str='./test_images/saved/'
# for img in coll:
#     if(img.shape[0]*img.shape[1]>=480*640):
#         if(i%10==0):
#             print(i)
#         io.imsave(name_str+str(i)+'.jpg',img)
#         i+=1
'''测试识别图片效果,清不清晰'''
# image_path='./test_images/saved/287.jpg'
# img=io.imread(image_path)
# fp.try_model0(img)


