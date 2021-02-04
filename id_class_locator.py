import numpy as np
import os
import sys
import shutil
import time
from cv2 import cv2
import time
import random
# This is needed since the notebook is stored in the object_detection folder.
from matplotlib import image, patches, pyplot as plt

path2File= os.path.dirname(os.path.realpath(__file__))

confidence=0.30
count=0

time_array=list()
docName=list()
#wakati=list()
result_var=list()

classes={ 0:'person',1:'bicycle',2: 'car',3:'motorcycle', 4:'airplance',5: 'bus',6:'train',7: 'truck', 8:'boat', 9:'traffic light', 10:'fire hydrant',\
11:'stop sign',12:'parking meter',13: 'bench',14:'bird', 15:'cat',16: 'dog',17:'horse',18: 'sheep', 19:'cow', 20:'elephant', 21:'bear', 22:'zebra'}

colors=['r','b','y','g','m', 'c', 'w']


def img_resize_aspect_ratio(imgArr, new_size):
    '''Resizing and keeping the aspect ratio'''
    h,w =imgArr.shape[:2]
    aspectRatio=h/w
    img=cv2.resize(imgArr,(new_size, int(new_size*aspectRatio)),  interpolation = cv2.INTER_AREA)
    return img

def cropped(imagePath, namer, coord):
    '''Crops the class from the frame'''
    img=cv2.imread(imagePath)
    print(coord)
    cropped_img=img[coord[1]:coord[3], coord[0]:coord[2]]
    cropped_img= img_resize_aspect_ratio(cropped_img, 1024)
    cv2.imwrite(path2File+'/static/output2/{}'.format(namer), cropped_img)


def id_class_detector(imageBuffer, model, namer, debug=False):
    '''Predicting the classes seen'''

    result_var, datalet, boxes=list(), list(), list()
    tempDict=dict()
     
    img1= img_resize_aspect_ratio(imageBuffer, 750)
   
    image_without_bbx=img1.copy()
    #t=time.time()
    # ## Download Model

    rows, cols= img1.shape[:2]
    
    model.setInput(cv2.dnn.blobFromImage(img1 , size=(320, 320), swapRB=True, crop=False))
   
    networkOutput = model.forward()
    fig, ax=plt.subplots(1)
    img2=cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    ax.imshow(img2)
    ax.axis("off")

    for i, detection in enumerate (networkOutput[0, 0]):

        score = float(detection[2])
        if score > confidence:

            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows

            rander=random.randint(0, 6)
            idx = int(detection[1])   # prediction class index.
        
            # draw the prediction on the frame
            label = "{}: {:.2f}%".format(classes[idx], score * 100)
          
            tempDict[str(classes[idx])]=[int(left), int(top), int(right), int(bottom)] 
            color=colors[i % len(colors)]
            
            rect= patches.Rectangle((left, top), (right - left), (bottom - top), linewidth=1, edgecolor=color, facecolor='none') 
            ax.text(left, top, label, fontsize=8, bbox={'facecolor':color, 'pad':4, 'ec':color})
            ax.add_patch(rect)
            ax.axis("off")
    
    #if os.path.isdir(path2File+'/home/adeola/Documents/FlaskApp3/static/input/{}'.format(namer)):
    plt.savefig(path2File+'/static/output/{}'.format(namer))
   
    

    
