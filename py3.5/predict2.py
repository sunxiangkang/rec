import tensorflow as tf
import cv2
import numpy as np

def Predict(imgs,pbModelPath):
    """
    with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        graphD=tf.GraphDef()
        with open(pbModelPath+'/model.pb','rb') as f:
            graphD.ParseFromString(f.read())
            tf.import_graph_def(graphD,name='')

        print("===out===:",sess.run(['FC21/add:0','FC22/add:0','FC23/add:0'],
                                        feed_dict={'inputs:0':imgs,'keepPorb:0':1.0}))
    """
    """
    with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        graphD = tf.GraphDef()
        with open ('./modelios/model.pb', 'rb') as f:
            graphD.ParseFromString(f.read())
            tf.import_graph_def(graphD, name='')

        print("~~~out~~~:",sess.run(['FC21/add:0','FC22/add:0','FC23/add:0'],
                                        feed_dict={'inputs:0':imgs}))
    """

    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, ['FC21/add','FC22/add','FC23/add'],'./modelios/modeliospb')
        print("+++out+++:",sess.run(['FC21/add:0','FC22/add:0','FC23/add:0'],
                       feed_dict={'inputs:0':imgs}))




if __name__=='__main__':
    imgPath='c:\\users\\Sunxk\\desktop\\test\\007.jpg'
    modelPath='D:\\Python\\Python35\\home\\rec\\model'
    img=cv2.imread(imgPath)
    inputTensor=np.array([img],np.float32)
    out=Predict(inputTensor,modelPath)