from recommend import recommend
from face_swap import swap_face

from skimage import io
import sys

#debug
import cv2

if __name__=='__main__':
    print("Welcome to the app")
    print("You can run the app with two arguments:\n\
            1. user image path\n\
            2. user gender")
    if(len(sys.argv)==1):
        print("You don't input any argument, we show the standard model.")
        user_img_path='test_images/100.jpg'
        user_gender=1
    else:
        user_img_path=sys.argv[1]
        user_gender=int(sys.argv[2])
    
    user_iamge_array=io.imread(user_img_path)
    L, user_face_type, style=recommend(user_gender,user_iamge_array)

    print("User gender: "+ str(user_gender))
    print("User face type: "+user_face_type)
    print("Recommend hair style :"+ style)
    print("Totally "+str(len(L))+" pictures")

    counter=0
    for img_path in L:
        print(counter)
        counter+=1

        io.imshow(swap_face(img_path,user_img_path))
        io.show()
        
        # img= swap_face(img_path,user_img_path)
        # cv2.imshow('img',img)
        # cv2.waitKey()