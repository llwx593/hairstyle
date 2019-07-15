# -*- coding: utf-8 -*-
import os
from skimage import io,transform
import numpy as np
import json

# from get_face_type import get_face_type
from predict_faceshape import predict_faceshape
from get_face_points import get_rotated_points_array
#debug
from face_swap import swap_face,swap_face_inner
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
    
    # recommend_vect=[] #final recommend 
    # index =0
    # for i in range(len(vect1)):
    #     recommend_vect.append(vect1[i]+ user_prefer_vector[i])
    #     if recommend_vect[i]>recommend_vect[index]:
    #         index=i
    index = list(range(len(vect1)))
    L = list(zip(vect1,index))              #[(0.3,1),...]
    L.sort(key = lambda x:x[0])     
    sorted_index = [e[1] for e in L]
    
    if user_gender==0:
        return [HAIR_MAN[index] for index in sorted_index]
    else:
        return [HAIR_WOMAN[index] for index in sorted_index]

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

    img_full_path_list=[]
    for img_name in img_name_list:
        img_full_path=os.path.join(style_dir_path,img_name)
        img_full_path_list.append(img_full_path)

    path_distance_list=[]

    for img_full_path in img_full_path_list:
        try:
            # print(counter)
            # counter=counter+1
            points_array=np.array(Dict[img_full_path])

            distance= np.mean(np.square(points_array-user_points_array))
            path_distance_list.append([img_full_path,distance])
        except:
            pass
    
    path_distance_list.sort(key=lambda x:x[1])
    return [e[0] for e in path_distance_list[:n]]

def random_code():
    s = ''
    for i in range(8):
        s += chr(48+np.random.randint(0,10))
    return s

def make_generated_img_dir(base64,gender):
    try:
        os.makedirs('output/0/')
        os.makedirs('output/1/')
    except:
        pass
    if(gender == 0):
        for style in HAIR_MAN:
            if(not os.path.exists('output/0/'+base64+'/'+style+'/')):
                os.makedirs('output/0/'+base64+'/'+style+'/')
    else:
        for style in HAIR_WOMAN:
            if(not os.path.exists('output/1/'+base64+'/'+style+'/')):
                os.makedirs('output/1/'+base64+'/'+style+'/')

def recommend(user_image_path, user_gender, user_prefer_vector=[0.5, 0.5, 0.5, 0.5, 0.5]):
    import time
    print(time.time())
    number = 3
    user_image_array = io.imread(user_image_path)
    user_face_type=predict_faceshape(user_gender,user_image_array)
    # user_face_type = '圆脸'
    print(user_face_type)
    print('get shape',time.time())

    user_points_array= get_rotated_points_array(user_image_array)

    style_list = recommend_hair(user_gender,user_face_type,user_prefer_vector)
    print('get style list',time.time())

    base64 = random_code() #用户编码
    make_generated_img_dir(base64,user_gender)
    print('gen dir',time.time())
    
    '''生成换脸效果图，并存储到相应目录里'''

    for style in style_list:
        if user_gender==0:
            style_dir_path= 'database/man/' + style +'/'
        else:
            style_dir_path= 'database/woman/' + style +'/'

        one_style_recomend_image_list = face_shape_sort(user_points_array, style_dir_path, n=number)
        print('face shape sort',time.time())

        counter=0

        path1_man ='output/0/'+base64+'/'+style+'/'
        path1_woman ='output/1/'+base64+'/'+style+'/'
        for img_path in one_style_recomend_image_list:
            #print(counter)
            counter+=1

            if (style==style_list[0]) and (img_path==one_style_recomend_image_list[0]):
                img, im2, landmarks2= swap_face(img_path,user_image_path)
            else:
                img = swap_face_inner(img_path,im2,landmarks2)

            if(user_gender==0):
                io.imsave(path1_man+str(counter)+'.jpg',img)
            else:
                io.imsave(path1_woman+str(counter)+'.jpg',img)
        print('one style gen pic',time.time())
        
    '''服务器返回json字符串'''
    #'{"number": 3,"faceshape": "圆脸","base64": "abcdefgijk","style": ["潮流","烫发","背头","短发","长发"]}'
    py_dic = {}
    py_dic['number']=number
    py_dic['faceshape'] = user_face_type
    py_dic['base64'] = base64
    py_dic['style'] = style_list
    
    return json.dumps(py_dic)

if __name__=='__main__':
    #print(random_code())
    
    #debug for recommend
    #girl
    user_img_path='test_images/100.jpg'
    user_img = io.imread(user_img_path)

    json_data =recommend(user_img_path,1)

    print(json_data)

    # #boy
    # user_img_path='test_images/52.jpg'
    # user_img = io.imread(user_img_path)

    # style_list,user_face_type =recommend(user_img_path, 0)

    # print("User face type: "+user_face_type)
    # print("recommend style"+str(style_list))

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
