# This is a sample Python script.
import cv2
from io import BytesIO
from PIL import Image
import base64
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import os
#import scipy
import tensorflow as tf
import webbrowser
import numpy as np
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# read the image
img = cv2.imread("checkagain.jpg")
print(type(img))
# convert BGR to RGB to be suitable for showing using matplotlib library
rimg = img.copy()
cimg = img.copy()
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# make a copy of the original image
imgdiv = []
# convert image to grayscale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# apply a blur using the median filter
img = cv2.medianBlur(img, 5)
# finds the circles in the grayscale image using the Hough transform
circles = cv2.HoughCircles(image=img, method=cv2.HOUGH_GRADIENT, dp=0.9,
                            minDist=80, param1=110, param2=39, minRadius=1, maxRadius=100)
print(type(circles))
circles = np.round(circles[0, :]).astype("int")
for (x, y, r) in circles:
    try:
        imgdiv.append(rimg[y-r-5:y+r+5, x-r-5:x+r+5])
        # draw the outer circle in green
        cv2.circle(cimg, (x, y), r, (0, 255, 0), 2)
        # draw the center of the circle in red
        cv2.circle(cimg, (x, y ), 2, (0, 0, 255), 3)
    except:
        print("Img not working")

# print the number of circles detected
print("Number of circles detected:", len(circles))
print(len(imgdiv))
# save the image, convert to BGR to save with proper colors
# cv2.imwrite("coins_circles_detected.png", cimg)
# show the image
plt.imshow(cimg)
plt.show()
f = 1
model = tf.keras.models.load_model("checkpoint_path")
dir_path = "C:/Users/aadip/PycharmProjects/Shape_Classifier/CreatedImages"
for e in imgdiv:
    try:
        imgName = './CreatedImages/'+str(f)+'.jpg'
        cv2.imwrite(imgName, e)
        plt.imshow(e)
        plt.show()
        print('d')
        print('d')
        print(type(e))
        #print(type(img))
        #im_b64 = base64.b64encode(e)
        #print('d')
        #im_bytes = base64.b64decode(im_b64)
        #print('d')
        #im_file = BytesIO(im_bytes)
        #PILImg = Image.open(im_file)
        PILImg = Image.fromarray(e)
        print('d')
        print(type(PILImg))
        PILImg.show()
        PILImg = PILImg.resize((75,75))
        #img = tf.keras.preprocessing.image.load_img(PILImg, target_size=(200,200))
        #img = tf.keras.preprocessing.image.load_img("C:/Users/aadip/PycharmProjects/Shape_Classifier/CreatedImages/"+str(f)+".jpg", target_size=(200,200))
        X = tf.keras.preprocessing.image.img_to_array(PILImg)
        #X = np.resize(X, (200, 200, 3))
        X = np.expand_dims(X, axis=0)
        images = np.vstack([X])
        val = model.predict(images)
        if val == 0:
            print('is a nickel')
        else:
            print('is a penny')
        f+=1
    except Exception as error:
        print(error)
        break
'''
#checkpoint_path = "cp.ckpt"
model = tf.keras.models.load_model("checkpoint_path")
dir_path = "C:/Users/aadip/PycharmProjects/Shape_Classifier/CreatedImages"
counter = 0
for i in os.listdir(dir_path):
    try:
        counter += 1
        print(i)
        img = tf.keras.preprocessing.image.load_img("C:/Users/aadip/PycharmProjects/Shape_Classifier/CreatedImages/"+str(counter)+".jpg", target_size=(200,200))
        X = tf.keras.preprocessing.image.img_to_array(img)
        X = np.expand_dims(X, axis=0)
        images = np.vstack([X])
        val = model.predict(images)
        if val == 0:
            print('is a nickel')
        elif val == 1:
            print('is a penny')
    except:
        break
'''
'''
for i in os.listdir(dir_path):
    try:
        os.remove('./CreatedImages/'+i)
    except: pass
'''
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
#C:\Users\aadip\PycharmProjects\Shape_Classifier\1.jpg