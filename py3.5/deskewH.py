import cv2
import numpy as np


def FitLineRansac(rectL,imgShape):
    ptsU=[];ptsB=[];count=0;Y=1
    for item in rectL:
        if len(item)>Y:
            count+=1
            ptsU.extend((temp[0],temp[1]) for temp in item)
            ptsB.extend((temp[0]+temp[2],temp[1]+temp[3]) for temp in item)

    if len(ptsU)>=2:
        if count>1:
            [vxU,vyU,xU,yU] = cv2.fitLine(np.array(ptsU), cv2.DIST_HUBER, 0, 0.01, 0.01)
            [vxB,vyB,xB,yB] = cv2.fitLine(np.array(ptsB), cv2.DIST_HUBER, 0, 0.01, 0.01)
            leftU = int((-xU  * vyU / vxU) + yU)
            rightU = int(((imgShape[1] - xU) * vyU / vxU) + yU)
            leftB = int((-xB * vyB / vxB) + yB)
            rightB = int(((imgShape[1] - xB) * vyB / vxB) + yB)
            return leftU,rightU,leftB,rightB
        else:
            rowNumU=int(sum(item[1] for item in ptsU)/len(ptsU))
            rowNumB=int(sum(item[1] for item in ptsB)/len(ptsB))
            return rowNumU,rowNumU,rowNumB,rowNumB
    return 0,0,0,0

def DeskewH(imgRGB,rectL):
    imgGray=cv2.cvtColor(imgRGB,cv2.COLOR_BGR2GRAY)
    imgH,imgW=imgGray.shape

    leftU, rightU, leftB, rightB = FitLineRansac(rectL, imgGray.shape)

    #testCode(imgRGB.copy(), leftU, rightU, leftB, rightB, "testcode")

    subImg = np.float32([[0, leftU], [imgW, rightU], [imgW, rightB], [0, leftB]])
    stdImg = np.float32([[0, 0], [imgW, 0], [imgW, int(imgH / 2)], [0, int(imgH / 2)]])
    mat = cv2.getPerspectiveTransform(subImg, stdImg)
    image = cv2.warpPerspective(imgRGB, mat, (imgW, int(imgH / 2)), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))

    return image

def testCode(img,leftU, rightU,leftB,rightB,winname):
    cv2.line(img,(0,leftU),(img.shape[1],rightU),(0,0,255),1)
    cv2.line(img,(0, leftB),(img.shape[1], rightB),(0, 0, 255),1)
    cv2.imshow(winname,img)
    cv2.waitKey()


if __name__=="__main__":
    from home.rec import crop
    imgPath = '.\\rotatetest\\6_3.png'
    img=cv2.imread(imgPath)
    rectL=crop.Crop(img,crop.condition1,crop.thres1)
    imgR=DeskewH(img,rectL)
    cv2.imshow("test",imgR)
    cv2.waitKey()