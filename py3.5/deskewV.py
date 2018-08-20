import cv2
import math
from home.rec import crop
import numpy as np
import scipy.ndimage.filters as f

Y=0

def Rot(img, angel):
    h,w=img.shape[:2]
    interval = abs( int( math.sin((float(angel) /180) * math.pi)* h))
    shapeN=h,w+interval
    pts1 = np.float32([[0,0],[0,h],[w,0],[w,h]])
    if(angel>0):
        pts2 = np.float32([[interval,0],[0,h],[shapeN[1],0 ],[w,h]])
    else:
        pts2 = np.float32([[0,0],[interval,h],[w,0],[shapeN[1],h]])
    M  = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,(shapeN[1],shapeN[0]),borderValue=(255,255,255))
    return dst

def DeskewV(imgRGB):
    imgRGBBordered=cv2.copyMakeBorder(imgRGB,int(imgRGB.shape[0]*0.1),int(imgRGB.shape[0]*0.1),0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
    area=np.zeros(51,np.int16)
    rectA=[]
    for theta in range(-25,25+1):
        if theta==0:
            tempImg=imgRGBBordered.copy()
        else:
            tempImg=Rot(imgRGBBordered,theta)
        rectL=crop.Crop(tempImg,crop.condition2,crop.thres2)
        dmax=-1;dmin=imgRGB.shape[1]
        U=0;B=0
        testImg=tempImg.copy()
        for item in rectL:
            if len(item)>Y:
                U=int(sum(rect[1] for rect in item)/len(item))
                B=int(sum(rect[1]+rect[3] for rect in item)/len(item))
                dL=max(rect[0] for rect in item)
                dR=min(rect[0]+rect[2] for rect in item)
                if dR>dmax:
                    dmax=dR
                if dL<dmin:
                    dmin=dL

        #testCode(testImg,dmax,dmin,"nidie")

        rectA.append((U,B,dmin,dmax))
        area[int(theta+25)]=dmax-dmin

    area = f.gaussian_filter1d(area,5)
    thetaB=int(area.argmin()-25)
    #print("竖直方向调整角度：",thetaB)
    imgOut=Rot(imgRGB,thetaB) if thetaB else imgRGB
    U,B,L,R=rectA[int(thetaB+25)]
    U=U if U-int(imgRGB.shape[0]*0.1)>0 else 0
    B=B if B<imgRGB.shape[0] else imgRGB.shape[0]
    imgOut=imgOut[U:B,L:R]
    return imgOut



def testCode(img,dmax,dmin,winName):
    cv2.line(img,(dmin,0),(dmin,img.shape[0]),(0,0,255),1)
    cv2.line(img,(dmax,0),(dmax,img.shape[1]),(0,0,255),1)
    cv2.imshow(winName,img)
    cv2.waitKey()


if __name__=="__main__":
    #from home.rec import crop
    from home.rec import deskewH
    imgPath = '.\\rotatetest\\test_7.jpg'
    img = cv2.imread(imgPath)
    rectL = crop.Crop(img, crop.condition1, crop.thres1)
    imgR = deskewH.DeskewH(img, rectL)
    out=DeskewV(imgR)
    cv2.imshow("out",out)
    cv2.waitKey()