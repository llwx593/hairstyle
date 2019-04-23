from recommend import face_shape_sort
from face_swap import swap_face

from skimage import io
import sys

if __name__=='__main__':
    print("Welcome to the app")
    print("You can run the app with two arguments:\n\
            1. user image path\n\
            2. style directory path.")
    if(len(sys.argv)==1):
        print("You don't input any argument, we show the standard model.")
        user_img_path='test_images/100.jpg'
        style_dir_path='test_images/girl/'
    else:
        user_img_path=sys.argv[1]
        style_dir_path=sys.argv[2]
    
    L,user_face_type=face_shape_sort(user_img_path,style_dir_path,20)

    print("User face type: "+user_face_type)
    print("Totally "+str(len(L))+" pictures")
    counter=0
    for img_path in L:
        print(counter)
        counter+=1

        io.imshow(swap_face(img_path,user_img_path))
        io.show()