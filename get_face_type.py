# -*- coding: utf-8 -*-
import numpy as np
from skimage import io
import os
import json

from get_face_points import get_rotated_points_array

FACE_NAME=['鹅蛋','方脸','心形','圆脸','长脸','钻石']

STANDARD_FACE_PATH_MAN='face_classify_old/standard_faces/man'
STANDARD_FACE_PATH_WOMAN='face_classify_old/standard_faces/woman'

FACE_TYPE=len(FACE_NAME)
SUB_FACE_TYPE_MAN = 3
SUB_FACE_TYPE_WOMAN=5

if os.path.exists('face_classify_old/standard_face_points_man.json'):
    with open('face_classify_old/standard_face_points_man.json') as file:
        L=json.load(file)
        STANDARD_FACE_POINTS_MAN=[]
        for one_shape_list in L:
            D=[]
            for p in one_shape_list:
                p=np.matrix(p)
                D.append(p)
            STANDARD_FACE_POINTS_MAN.append(D)
else:
    print('face_classify_old/standard_face_points_man.json not found')
#woman
if os.path.exists('face_classify_old/standard_face_points_woman.json'):
    with open('face_classify_old/standard_face_points_woman.json') as file:
        L=json.load(file)
        STANDARD_FACE_POINTS_WOMAN=[]
        for one_shape_list in L:
            D=[]
            for p in one_shape_list:
                p=np.matrix(p)
                D.append(p)
            STANDARD_FACE_POINTS_WOMAN.append(D)
else:
    print('face_classify_old/standard_face_points_woman.json not found')

def get_face_type(gender, img_array):
    if gender == 0: 
        points_array=get_rotated_points_array(img_array)

        d=[]
        for i in range(FACE_TYPE):
            for j in range(SUB_FACE_TYPE_MAN):
                temp=np.mean(np.square(STANDARD_FACE_POINTS_MAN[i][j]-points_array) )
                #print(temp)
                d.append(temp)
        m=0
        for i in range(FACE_TYPE*SUB_FACE_TYPE_MAN):
            if(d[i]<d[m]):
                m=i
        return FACE_NAME[m//SUB_FACE_TYPE_MAN]
    else:
        points_array=get_rotated_points_array(img_array)

        d=[]
        for i in range(FACE_TYPE):
            for j in range(SUB_FACE_TYPE_WOMAN):
                temp=np.mean(np.square(STANDARD_FACE_POINTS_WOMAN[i][j]-points_array) )
                #print(temp)
                d.append(temp)
        m=0
        for i in range(FACE_TYPE*SUB_FACE_TYPE_WOMAN):
            if(d[i]<d[m]):
                m=i
        return FACE_NAME[m//SUB_FACE_TYPE_WOMAN]

if __name__=='__main__':
    path='test_images/man'
    img_path_list=os.listdir(path)
    
    counter=0
    for img_path in img_path_list:
        print(counter,len(img_path_list))
        counter+=1
        img=io.imread(os.path.join(path,img_path))
        
        face_type=get_face_type(1, img)
        print(face_type)

        io.imshow(img)
        io.show()


        #io.imsave(os.path.join('face_classify_old/'+face_type+'/', img_path),img)

