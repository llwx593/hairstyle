import face_points as fp
from face_points import NoFaceException

import numpy as np
from skimage import io
import json

def get_face_attributes_list(image_array):
    '''输入值域为（0，255）的图片像素矩阵，返回4个属性的列表，
    前3个属性都是数值，第4个属性是ndarray'''
    points_array = fp.get_points_array_with_image_array(image_array)
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
    variance_sqrt=np.array([[3.73735969, 0.27621047, 0.08418358, 0.10745149]]).reshape(4,1)
    mean=np.array([[ 0.05842521, -0.00543055,  0.00099243,  0.33862748]]).reshape(4,1)
    distance_array=(distance_array-mean)/variance_sqrt
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
user_face_image_path='images\\women\\2.jpg'
#path='test_images\\student_man\\*.jpg'
#path='images\\man\\*.jpg'
path='images\\women\\*.jpg'
collections=io.ImageCollection(path)
collections_user=io.ImageCollection(user_face_image_path)

user_face_image=io.imread(user_face_image_path)
user_face_attribute_list=get_face_attributes_list(user_face_image)
result = []
count=0
for image in collections:
    if(count%10==0):
        print(count)
    count+=1
    try:
        attribute_list= get_face_attributes_list(image)
        result.append( [ compare_face(user_face_attribute_list, attribute_list) , image,
            get_4_distance_and_norm(user_face_attribute_list,attribute_list )**2 ] )
    except:
        pass
result.sort()

for i in range(len(result)):
    e=result[i]
    print('=================',i,'===================')
    i+=1
    print(e[0], e[2].reshape(1,-1))

    io.imshow(e[1])
    io.show()
'''获得调节4个属性向量的均值与方差'''
# #获得图片
# path='./test_images/saved/*.jpg'
# coll=io.ImageCollection(path)
# #获得所有图片的属性列表
# attributes=[]
# count=0
# for image in coll[500:1500]:
#     try:
#         attributes.append(get_face_attributes_list(image))
#         if(count%10==0):
#                 print(count)
#     except NoFaceException as e:
#         print("未探测到脸,图片编号："+str(count))
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
# mean=np.sum(Result,axis=0,keepdims=True)/count
# variance=np.sum((Result-mean)**2,axis=0,keepdims=True)/count
# variance_sqrt=np.sqrt(variance)
# print(count)
# print(mean)
# print(variance_sqrt)
# '''
# :1000
# 498501
# [[ 0.36320478 -0.02352534  0.00160396  0.33969678]]
# [[3.62749238 0.30238229 0.08054489 0.10833611]]

# 1000:
# 566580
# [[-0.0155081  -0.00613732  0.00053208  0.34195277]]
# [[3.79367141 0.26486188 0.08769826 0.10836627]]

# 500:1500
# 499500
# [[-0.16188268  0.01342973  0.0009043   0.33378846]]
# [[3.78313319 0.26296363 0.0838283  0.10553102]]
# '''
'''筛选出像素高的图片'''
#path='./test_images/women/*.jpg'
#coll=io.ImageCollection(path)
#i=0
#name_str='./test_images/saved/'
#for img in coll:
#    if(img.shape[0]*img.shape[1]>=480*640):
#        if(i%10==0):
#            print(i)
#        io.imsave(name_str+str(i)+'.jpg',img)
#        i+=1
'''测试识别图片效果,清不清晰'''
# image_path='./test_images/saved/632.jpg'
# img=io.imread(image_path)
# fp.try_model0(img)


