# -*- coding: utf-8 -*-
import numpy as np
from skimage import io
import os

from get_face_points import get_rotated_points_array

FACE_NAME=['鹅蛋','方脸','心形','圆脸','长脸','钻石']
STANDARD_FACE_PATH='face_classification/standard_faces/'

FACE_TYPE=len(FACE_NAME)
SUB_FACE_TYPE=5

STANDARD_FACE_POINTS_ARRAY=[]
for i in range(FACE_TYPE):
    e=[]
    face_path=os.path.join(STANDARD_FACE_PATH,FACE_NAME[i])
    file_name_list=os.listdir(face_path)

    for file_name in file_name_list:
        img=io.imread(os.path.join(face_path,file_name))
        e.append(get_rotated_points_array(img))

    STANDARD_FACE_POINTS_ARRAY.append(e)

def get_face_type(img_array):
    points_array=get_rotated_points_array(img_array)

    d=[]
    for i in range(FACE_TYPE):
        for j in range(SUB_FACE_TYPE):
            temp=np.mean(np.square(STANDARD_FACE_POINTS_ARRAY[i][j]-points_array) )
            #print(temp)
            d.append(temp)
    m=0
    for i in range(FACE_TYPE*SUB_FACE_TYPE):
        if(d[i]<d[m]):
            m=i
    return FACE_NAME[m//SUB_FACE_TYPE]

if __name__=='__main__':
    for i in range(FACE_TYPE):
        if not os.path.exists('face_classification/'+FACE_NAME[i]+'/'):
            os.mkdir('face_classification/'+FACE_NAME[i]+'/')

    path='student_image/'
    img_path_list=os.listdir(path)

    counter=0
    for img_path in img_path_list:
        print(counter,len(img_path_list))
        counter+=1
        img=io.imread(os.path.join(path,img_path))

        face_type=get_face_type(img)
        print(face_type)

        io.imshow(img)
        io.show()


        #io.imsave(os.path.join('face_classification/'+face_type+'/', img_path),img)

