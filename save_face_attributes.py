# -*- coding: utf-8 -*-

from get_face_attributes import get_attributes_list,NoFaceException,TooManyFaces
from skimage import io
import json
import os

def save_attributes_as_json(database_path):
    json_file_path=os.path.join(database_path,'data.json')

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
            except NoFaceException or TooManyFaces:
                pass
    with open(json_file_path,'w') as json_file:
        json.dump(data_dic,json_file)
        print('存储成功')
if __name__=="__main__":
    save_attributes_as_json('database/')