# -*- coding: utf-8 -*-
import os
from skimage import io
import numpy as np
import json

from get_face_type import get_face_type
from get_face_points import get_rotated_points_array
#debug
from face_swap import swap_face

def load_json(style_dir_path):
#得到数据库中的图片的脸型并存储
    data_path=os.path.join(style_dir_path,'data.json')
    if os.path.exists(data_path):
        with open(data_path) as file:
            Dict=json.load(file)
            return Dict
    else:
        Dict={}
        img_name_list=os.listdir(style_dir_path)
        counter=0
        print('totally: '+str(len(img_name_list)))
        for img_name in img_name_list:
            try:
                print(counter)
                counter+=1
                img_full_path=os.path.join(style_dir_path,img_name)
                #Dict[img_full_path]={}

                img=io.imread(img_full_path)
                shape=get_face_type(img)

                Dict[img_full_path]=shape
            except:
                pass
        with open(data_path,'w') as file:
            json.dump(Dict,file)
            return Dict

def face_shape_sort(user_img_path,style_dir_path,n=10):
    user_img=io.imread(user_img_path)
    user_face_type=get_face_type(user_img)
    user_points_array=get_rotated_points_array(user_img)

    Dict=load_json(style_dir_path)

    L=[]
    img_name_list=os.listdir(style_dir_path)

    for img_name in img_name_list:
        try:
            img_full_path=os.path.join(style_dir_path,img_name)

            shape=Dict[img_full_path]
            if(shape==user_face_type):
                L.append(img_full_path)
        except:
            pass
    D=[]
    for img_full_path in L:
        img=io.imread(img_full_path)
        points_array=get_rotated_points_array(img)

        distance= np.mean(np.square(points_array-user_points_array))
        D.append([img_full_path,distance])

    D.sort(key=lambda x:x[1])
    return [e[0] for e in D],user_face_type

if __name__=='__main__':
    user_img_path='test_images/22.jpg'
    style_dir_path='test_images/girl/'
    L,user_face_type=face_shape_sort(user_img_path,style_dir_path,20)

    print("User face type: "+user_face_type)
    print("Totally "+str(len(L))+" pictures")
    counter=0
    for img_path in L:
        print(counter)
        counter+=1

        io.imshow(swap_face(img_path,user_img_path))
        io.show()

