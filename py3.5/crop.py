import cv2
import os
import math
import numpy as np

condition1=[0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9,0.05,0.6,0.35,0.55]
thres1=[75,150,20]
condition2=[0.1,0.9,0,1,0.1,0.9,0,1,0.01,0.6,0.7,1]
thres2=[75,150,10]
Y=1

def Crop(imgRGB,condition,y):
    imgGray=cv2.cvtColor(imgRGB,cv2.COLOR_BGR2GRAY)
    imgGray=cv2.GaussianBlur(imgGray,(5,5),0)

    imgH,imgW=imgGray.shape

    rectL=[]
    for thres in np.linspace(y[0],y[1],y[2]):
        testImg=imgRGB.copy()
        #imgThres=cv2.adaptiveThreshold(imgGray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,5,thres)
        _,imgThres=cv2.threshold(imgGray,thres,255,cv2.THRESH_BINARY)
        #cv2.imshow("imgThres",imgThres)
        #cv2.waitKey()
        reImg,contours,hierarchy=cv2.findContours(imgThres.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        #print("len(contours):",len(contours))
        for cont in contours:
            x,y,w,h=cv2.boundingRect(cont)

            #testCode(testImg,(x,y,w,h),"test")

            if imgW*condition[0]<x and x<imgW*condition[1] \
                and imgH*condition[2]<y and y<imgH*condition[3]\
                and imgW*condition[4]<(x+w) and (x+w)<imgW*condition[5]\
                and imgH*condition[6]<(y+h) and (y+h)<imgH*condition[7]\
                and imgW*condition[8]<w and w<imgW*condition[9] \
                and imgH * condition[10] < h and h < imgH * condition[11]:

                #testCode(testImg,(x,y,w,h),"test")

                if len(rectL) == 0:
                    rectL.append([(x,y,w,h)])
                else:
                    for i in range(len(rectL)):
                        if math.fabs(x - rectL[i][0][0]) < 0.3 * rectL[i][0][2] \
                            and math.fabs(w- rectL[i][0][2]) < 0.3 * rectL[i][0][2]:
                            rectL[i].append((x,y,w,h))
                            break
                        elif i == len(rectL) - 1:
                            rectL.append([(x,y,w,h)])
    return rectL

def CropImg(img,rectL):
    xU=999;yU=999
    xL=-1;yL=-1
    for item in rectL:
        if len(item)>Y:
            for rect in item:
                x, y, w, h = rect
                if x<xU:
                    xU=x
                if y<yU:
                    yU=y
                if x+w>xL:
                    xL=x+w
                if y+h>yL:
                    yL=y+h
    return img[yU:yL,xU:xL]


def testCode(img,rect,winname):
    x,y,w,h=rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
    cv2.imshow(winname,img)
    cv2.waitKey()


if __name__=="__main__":
    """
    for parent,dirs,filenames in os.walk('C:\\Users\\Sunxk\\Desktop\\temp'):
        subdir='C:\\Users\\Sunxk\\Desktop\\temp_sub'
        for file in filenames:
            imgpath=os.path.join(parent,file)
            img=cv2.imread(imgpath)
            rect=Crop(img,condition1,thres1)
            kk=CropImg(img,rect)
            if kk.shape[0]>0:
                kk=cv2.resize(kk,(64, 54),interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(os.path.join(subdir,file),kk)
    """
    #imgPath='C:\\Users\\Sunxk\\Desktop\\subImg\\test_7.jpg'
    imgPath='.\\rotatetest\\test_6.jpg'
    img=cv2.imread(imgPath)
    rectL=Crop(img,condition1,thres1)
    Y=1
    for item in rectL:
        if len(item)>Y:
            for rect in item:
                x, y, w, h = rect
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                cv2.imshow("out",img)
                cv2.waitKey()