import tensorflow as tf
import cv2
import numpy as np
import os
from tensorflow.python.framework import graph_util

def freezeGraph(img,ckptModelPath):
    ckpt=tf.train.get_checkpoint_state(ckptModelPath)
    if ckpt and ckpt.model_checkpoint_path:
        saver=tf.train.import_meta_graph(ckpt.model_checkpoint_path+'.meta',clear_devices=True)

        graph=tf.get_default_graph()
        graphD=graph.as_graph_def()

        with tf.Session() as sess:
            saver.restore(sess,ckpt.model_checkpoint_path)
            print('===out===:',sess.run(['FC21/add:0','FC22/add:0','FC23/add:0'],
                                        feed_dict={'inputs:0':img}))


            constGraph=graph_util.convert_variables_to_constants\
                (sess,graphD,['FC21/add','FC22/add','FC23/add'])
            with tf.gfile.FastGFile(ckptModelPath+'/model.pb','wb') as f:
                f.write(constGraph.SerializeToString())

            builder = tf.saved_model.builder.SavedModelBuilder('./modelios/modeliospb')
            builder.add_meta_graph_and_variables(sess, ['FC21/add','FC22/add','FC23/add'])
            builder.save()


if __name__=='__main__':
    imgPath='c:\\users\\Sunxk\\desktop\\test\\145.jpg'
    ckptModelPath='D:\\Python\\Python35\\home\\rec\\model'
    img=cv2.imread(imgPath)
    inputTensor=np.array([img],dtype=np.float32)
    freezeGraph(inputTensor, ckptModelPath)