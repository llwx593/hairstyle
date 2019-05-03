# -*- coding: utf-8 -*-
import os
from skimage import io
import numpy as np
import json

from get_face_type import get_face_type
from get_face_points import get_rotated_points_array
#debug
from face_swap import swap_face
import cv2

HAIR_MAN=['短发','长发','背头','烫发','潮流']
HAIR_WOMAN=['短发','长发','偏分','束发','潮流']

SHAPE_TO_HAIR_DICT_MAN=\
{
    '方脸':[0.4, 0.2, 0.5, 0.3, 0.3],
    '长脸':[0.2, 0.5, 0.4, 0.3, 0.3],
    '鹅蛋':[0.5, 0.5, 0.5, 0.5, 0.5],
    '圆脸':[0.4, 0.2, 0.3, 0.3, 0.3],
    '心形':[0.1, 0.4, 0.5, 0.3, 0.3],
    '钻石':[0.2, 0.4, 0.4, 0.3, 0.3]
}
SHAPE_TO_HAIR_DICT_WOMAN=\
{
    '方脸':[0.2, 0.4, 0.5, 0.2, 0.3],
    '长脸':[0.2, 0.5, 0.4, 0.3, 0.3],
    '鹅蛋':[0.5, 0.5, 0.5, 0.5, 0.5],
    '圆脸':[0.4, 0.2, 0.5, 0.3, 0.3],
    '心形':[0.3, 0.4, 0.4, 0.5, 0.3],
    '钻石':[0.5, 0.5, 0.3, 0.3, 0.3]
}

def recommend_hair(user_gender,user_face_type,\
                user_prefer_vector):
    # user_gender: 0 for man, 1 for woman
    if user_gender==0:
        vect1= SHAPE_TO_HAIR_DICT_MAN[user_face_type]
    else:
        vect1= SHAPE_TO_HAIR_DICT_WOMAN[user_face_type]
    
    recommend_vect=[] #final recommend 
    index =0
    for i in range(len(vect1)):
        recommend_vect.append(vect1[i]+ user_prefer_vector[i])
        if recommend_vect[i]>recommend_vect[index]:
            index=i
    if user_gender==0:
        return HAIR_MAN[index]
    else:
        return HAIR_WOMAN[index]
        
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
                p = get_rotated_points_array(img)
                p=np.array(p)
                
                Dict[img_full_path]=[list(e) for e in list(p)]
            except:
                pass
        with open(data_path,'w') as file:
            json.dump(Dict,file)
            return Dict

def face_shape_sort(user_points_array,style_dir_path,n=10):
    Dict=load_json(style_dir_path)

    img_name_list=os.listdir(style_dir_path)

    L=[]
    for img_name in img_name_list:
        img_full_path=os.path.join(style_dir_path,img_name)
        L.append(img_full_path)

    D=[]

    for img_full_path in L:
        try:
            # print(counter)
            # counter=counter+1
            points_array=np.array(Dict[img_full_path])

            distance= np.mean(np.square(points_array-user_points_array))
            D.append([img_full_path,distance])
        except:
            pass
    
    D.sort(key=lambda x:x[1])
    return [e[0] for e in D[:n]]

def recommend(user_gender, user_image_array, user_prefer_vector=[0.5, 0.5, 0.5, 0.5, 0.5]):
    user_face_type=get_face_type(user_gender,user_image_array)
    user_points_array= get_rotated_points_array(user_image_array)

    style = recommend_hair(user_gender,user_face_type,user_prefer_vector)

    if user_gender==0:
        style_dir_path= 'database/man/' + style +'/'
    else:
        style_dir_path= 'database/woman/' + style +'/'

    L = face_shape_sort(user_points_array, style_dir_path)
    return L,user_face_type, style

if __name__=='__main__':
    #debug for recommend
    user_img_path='test_images/100.jpg'
    user_img = io.imread(user_img_path)

    L,user_face_type , style =recommend(1, user_img)

    print("User face type: "+user_face_type)
    print("Totally "+str(len(L))+" pictures")
    print(L,user_face_type, style)

    #debug for face_shape_sort

    # user_img_path='test_images/58.jpg'
    # style_dir_path= 'test_images/man/'

    # # user_img_path='test_images/100.jpg'
    # # style_dir_path= 'test_images/best/woman/'

    # user_img = io.imread(user_img_path)
    # user_points_array= get_rotated_points_array(user_img)

    # L= face_shape_sort(user_points_array,style_dir_path,100)
    # counter=0
    # for img_path in L:
    #     print(counter)
    #     counter+=1

    #     # img= swap_face(img_path,user_img_path)
    #     # io.imshow(img)
    #     # io.show()

    #     img = swap_face(img_path, user_img_path)
    #     #img = cv2.resize(img, (int(img.shape[1]/1.5),int(img.shape[0]/1.5)))

    #     img =cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #     # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    #     cv2.imshow('img',img)
    #     cv2.waitKey()


    # #before
    # #debug for face_shape_sort 
    # # user_img_path='test_images/22.jpg'
    # # style_dir_path= 'test_images/man/'

    # user_img_path='test_images/100.jpg'
    # style_dir_path= 'test_images/woman/'

    # user_img = io.imread(user_img_path)
    # user_points_array= get_rotated_points_array(user_img)

    # user_face_type = get_face_type(user_img)
    # L = face_shape_sort(user_face_type,user_points_array,style_dir_path)

    # print('Totally :'+str(len(L)))
    # print('User face shape: '+user_face_type)
    # counter=0
    # for img_path in L:
    #     print(counter)
    #     counter+=1

    #     # img= swap_face(img_path,user_img_path)
    #     # io.imshow(img)
    #     # io.show()

    #     img = swap_face(img_path, user_img_path)
    #     #img = cv2.resize(img, (int(img.shape[1]/1.5),int(img.shape[0]/1.5)))

    #     img =cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #     # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    #     cv2.imshow('img',img)
    #     cv2.waitKey()
