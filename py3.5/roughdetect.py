import cv2
import cache

def RoughDetect(imgRGB):
    imgGray=cv2.cvtColor(imgRGB,cv2.COLOR_BGR2GRAY)

    modelPath='./model/cascademodel/cascade.xml'
    watch_cascade = cv2.CascadeClassifier(modelPath)
    watches = watch_cascade.detectMultiScale(imgGray, 1.1, 1)

    subImg=[]
    for (x, y, w, h) in watches:
        tempImg=imgRGB[y:y+h+1,x:x+w+1]
        subImg.append(tempImg)
    return subImg


if __name__=="__main__":
    import sys
    sys.path.append('.')
    imgPath=r'./testimgs/2.jpg'
    img=cv2.imread(imgPath)
    subImgs=RoughDetect(img)
    for subImg in subImgs:
        cv2.imshow("test",subImg)
        cv2.waitKey()
        cache.Cache(subImg)


    """
    path='c:\\users\\Sunxk\\desktop\\test.jpg'
    imgRGB=cv2.imread(path)

    dirPath=os.path.dirname(path)
    fileName=os.path.basename(path)

    subImgPath=os.path.join(dirPath,"subImg")
    if not os.path.exists(subImgPath):
        os.makedirs(subImgPath)
    for parent,dirs,files in os.walk(subImgPath):
        for file in files:
            if file.startswith(fileName.split('.')[0]):
                os.remove(os.path.join(parent,file))

    subImg=RoughDetect(imgRGB)
    for i in range(len(subImg)):
        subImgName = os.path.join(subImgPath, fileName.replace(".jpg", "_" + str(i+1) + ".jpg"))
        cv2.imwrite(subImgName,subImg[i])

    ---------------------------------------------------------------------------------------------
    path='c:\\users\\Sunxk\\desktop\\img'
    dpath='c:\\users\\Sunxk\\desktop\\temp'
    for parents,dirs,files in os.walk(path):
        for file in files:
            absPath=os.path.join(parents,file)
            img=cv2.imread(absPath)

            subImgs=RoughDetect(img)
            for i in range(len(subImgs)):
                subImgName = os.path.join(dpath, file.replace(".jpg", "_" + str(i + 1) + ".jpg"))
                cv2.imwrite(subImgName, subImgs[i])
    """