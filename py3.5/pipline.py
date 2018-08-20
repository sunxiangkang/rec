import roughdetect
import cache
import crop
import deskewH
import deskewV
import predict
import tensorflow as tf
import argparse
import cv2

def Main(imgPath,cached=True):
    imgRGB=cv2.imread(imgPath)
    subImgs=roughdetect.RoughDetect(imgRGB)
    sess=tf.Session()
    pred=predict.Predict(sess,'.\\model')
    res=[]
    for subImg in subImgs:
        if cached:
            cache.Cache(subImg)
        rectL=crop.Crop(subImg,crop.condition1,crop.thres1)
        subImgDeskewH=deskewH.DeskewH(subImg,rectL)
        imgRes=deskewV.DeskewV(subImgDeskewH)
        res.append(pred.predict(imgRes))
    sess.close()
    return max(res)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--imagepath",default='.\\testimgs\\1.jpg',type=str)
    parser.add_argument("--cached", default=True, type=bool)
    args = parser.parse_args()
    res=Main(args.imagepath,args.cached)
    print("out:",res)