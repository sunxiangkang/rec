import tensorflow as tf
from home.rec import model
import numpy as np
import cv2


class Predict:
    def __init__(self,sess,modelPath):
        self.sess=sess
        self.modelPath=modelPath
        self.logit1,self.logit2,self.logit3=self.__LoadModel()

    def __LoadModel(self):
        imgsHolder = tf.placeholder(dtype=tf.float32, shape=[1, 54, 64, 3],name='imgholder')
        keepHolder = tf.placeholder(tf.float32, name="keepProb")
        logits1, logits2, logits3 = model.Inference(imgsHolder, keepHolder)
        saver = tf.train.Saver()
        ckp = tf.train.get_checkpoint_state(self.modelPath)
        if ckp and ckp.model_checkpoint_path:
            saver.restore(self.sess, ckp.model_checkpoint_path)
            return logits1,logits2,logits3
        else:
            raise Exception("fail to load model,check modelPath.")

    def predict(self,img):
        imgReshape=cv2.resize(img,(64,54))
        inputTensor = np.array([imgReshape])
        imgsHolder=tf.get_default_graph().get_tensor_by_name("imgholder:0")
        keepHolder=tf.get_default_graph().get_tensor_by_name("keepProb:0")
        res1, res2, res3 = self.sess.run([self.logit1, self.logit2, self.logit3], feed_dict={imgsHolder: inputTensor, keepHolder: 1.0})
        prediction = np.reshape(np.array([res1, res2, res3]), [-1, 10])
        maxIndex = np.argmax(prediction, axis=1)
        out = int(''.join(map(str, maxIndex)))

        print("识别结果：", out)

        return out


    """
    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckp=tf.train.get_checkpoint_state(path)
        if ckp and ckp.model_checkpoint_path:
            saver.restore(sess,ckp.model_checkpoint_path)
            res1,res2,res3=sess.run([logits1,logits2,logits3],feed_dict={imgsHolder:img,keepHolder:1.0})
            print('===out===:',res1,res2,res3)
            prediction=np.reshape(np.array([res1,res2,res3]),[-1,10])
            maxIndex=np.argmax(prediction,axis=1)
            out=int(''.join(map(str,maxIndex)))

            print("识别结果：",out)

            return out
        else:
            return False
    """

if __name__=='__main__':
    imgPath='.\\testimgs\\predtest\\007_10.png'
    modelPath='.\\model'
    sess=tf.Session()
    img=cv2.imread(imgPath)
    pred=Predict(sess,modelPath)
    out=pred.predict(img)
