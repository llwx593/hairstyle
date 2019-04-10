from get_face_attributes import get_attributes_list,NoFaceException,TooManyFaces

import math
import os
import numpy as np
from skimage import io
import json


def diff(a,b):
    return math.fabs((a-b)/b)

def get_data_json(database_path):
    '''
    传入数据库文件夹路径，返回该文件夹下所有图片的一个字典，key为图片路径字符串，
    value为图片脸型的属性列表
    '''
    json_file_path=os.path.join(database_path,'data.json')
    if os.path.exists(json_file_path):
        with open(json_file_path) as json_file:
            return json.load(json_file)
    else:
        data_dic={}
        image_path_list=os.listdir(database_path)
        print('总共%d张图片'%(len(image_path_list)))
        counter=0
        for image_path in image_path_list:
            image_path=os.path.join(database_path,image_path)
            if image_path.split('.')[1]=='jpg' or image_path.split('.')[1]=='png':
                image_array=io.imread(image_path)
                try:
                    if(image_array.shape[2]==4):
                        image_array=image_array[:,:,:3]
                    data_dic[image_path]=get_attributes_list(image_array)
                    if(counter%10==0):
                        print('已处理第%d张图片'%counter)
                    counter+=1
                except NoFaceException:
                    pass
        with open(json_file_path,'w') as json_file:
            json.dump(data_dic,json_file)
            print('存储成功')
        return data_dic

#0.012 0.06-0.08 [0.02,0.07,0.1,0.05]
def filter_debug(user_image_path,database_path,W=[0.05,0.1,0.2,0.1],debug=True):
    user_image_array=io.imread(user_image_path)
    try:
        user_attribute_list=get_attributes_list(user_image_array)
    except NoFaceException:
        print('未检测到人脸')
        return 0
    except TooManyFaces:
        print('图片含有多张人脸')
        return 0
    data_dic=get_data_json(database_path)

    if debug:
        result=[]
        for image_path,attributes_list in data_dic.items():
            result.append([attributes_list,image_path])
        fiter_1=[]
        for e in result:
            if(diff(e[0][0],user_attribute_list[0])<W[0]):
                fiter_1.append(e)
        fiter_2=[]
        for e in fiter_1:
            if(diff(e[0][1],user_attribute_list[1])<W[1]):
                fiter_2.append(e)
        fiter_3=[]
        for e in fiter_2:
            if(diff(e[0][2],user_attribute_list[2])<W[2]):
                fiter_3.append(e)
        fiter_4=[]
        for e in fiter_3:
            if(diff(e[0][3],user_attribute_list[3])<W[3]):
                fiter_4.append(e)
        if(debug):
            print(len(fiter_1))
            print(len(fiter_2))
            print(len(fiter_3))
            print(len(fiter_4))
        for i in range(len(fiter_4)):
            e=fiter_4[i]
            print('=================',i,'===================')
            i+=1
            print(user_attribute_list)
            print(e[0])
            differ=[]
            for i in range(len(user_attribute_list)):
                differ.append( diff(e[0][i],user_attribute_list[i]) )
            print(differ)
            io.imshow(io.imread(e[1]))
            io.show()
            #fp.try_model(io.imread(e[1]))
            #fp.try_model(user_image_array)

def filter(user_image_path,database_path,W=[0.02,0.07,0.1,0.05]):
    '''从数据库中筛选脸型较为相似的图片，返回图片的路径列表'''
    user_image_array=io.imread(user_image_path)
    try:
        user_attribute_list=get_attributes_list(user_image_array)
    except NoFaceException:
        print('未检测到人脸')
        return 0
    except TooManyFaces:
        print('图片含有多张人脸')
        return 0
    data_dic=get_data_json(database_path)

    result=[]
    for image_path,attributes_list in data_dic.items():
        for i in range(4):
            if diff(attributes_list[i],user_attribute_list[i])>=W[i]:
                break
        else:
            result.append([diff(attributes_list[0],user_attribute_list[0]),image_path])
    result.sort()

    image_paths=[e[1] for e in result]

    return image_paths

#student_images/student_man/
#test_images/man/
#user_image_path='test_images/22.jpg'
#database_path='student_images/student_man/'
#filter(user_image_path,database_path)


