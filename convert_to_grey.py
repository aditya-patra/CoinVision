import os
import cv2
dir_path = '.\Coin_Img\Testing'
for i in os.listdir(dir_path):
    print(str(dir_path+"//"+i))
    img = cv2.imread(os.path.join(dir_path, i))
    greyscale = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(os.path.join(dir_path, i), greyscale)
dir_path = '.\Coin_Img\Training'
for i in os.listdir(dir_path):
    for e in os.listdir(os.path.join(dir_path, i)):
        print(os.path.join(dir_path, i, e))
        img = cv2.imread(os.path.join(dir_path, i, e))
        greyscale = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(os.path.join(dir_path, i, e), greyscale)
dir_path = '.\Coin_Img\Validation'
for i in os.listdir(dir_path):
    for e in os.listdir(os.path.join(dir_path, i)):
        print(os.path.join(dir_path, i, e))
        img = cv2.imread(os.path.join(dir_path, i, e))
        greyscale = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(os.path.join(dir_path, i, e), greyscale)
        print(img.shape)
        #cv2.imshow('img', greyscale)
        #cv2.waitKey(5000)