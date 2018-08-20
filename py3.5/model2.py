import tensorflow as tf
import numpy as np
import datetime
import os
import cv2


def GenerateWegiths(wName,wShape,wDtype=tf.float32,wInitiallizer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32)):
    weights=tf.get_variable(wName,shape=wShape,dtype=wDtype,initializer=wInitiallizer)

    return weights

def GenerateBias(bName,bShape,bDtype=tf.float32,bInitializer=tf.constant_initializer(0.1)):
    bias=tf.get_variable(bName,shape=bShape,dtype=bDtype,initializer=bInitializer)

    return bias


def Inference(imgs,keepProb):
    imgs=tf.cast(imgs,tf.float32)
    with tf.variable_scope("conv1") as scope:
        weights = GenerateWegiths('weights',[3,3,3,32])
        conv=tf.nn.conv2d(imgs,weights,strides=[1,1,1,1],padding='VALID')
        biases = GenerateBias('biases',[32])
        preActivition=tf.nn.bias_add(conv,biases)
        conv1=tf.nn.relu(preActivition,name=scope.name)

    with tf.variable_scope("conv2") as scope:
        weights = GenerateWegiths('weights',[3,3,32,32])
        conv=tf.nn.conv2d(conv1,weights,strides=[1,1,1,1],padding='VALID')
        biases = GenerateBias('biases',[32])
        preActivition=tf.nn.bias_add(conv,biases)
        conv2=tf.nn.relu(preActivition,name=scope.name)

    with tf.variable_scope("MaxPool1") as scope:
        pool1=tf.nn.max_pool(conv2,[1,2,2,1],strides=[1,2,2,1],padding='VALID',name='pooling1')

    with tf.variable_scope("conv3") as scope:
        weights = GenerateWegiths('weights',[3,3,32,64])
        conv = tf.nn.conv2d(pool1, weights, strides=[1, 1, 1, 1], padding='VALID')
        biases = GenerateBias('biases',[64])
        preActivition = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(preActivition, name=scope.name)

    with tf.variable_scope("conv4") as scope:
        weights = GenerateWegiths('weights',[3,3,64,64])
        conv = tf.nn.conv2d(conv3, weights, strides=[1, 1, 1, 1], padding='VALID')
        biases = GenerateBias('biases',[64])
        preActivition = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(preActivition, name=scope.name)

    with tf.variable_scope("MaxPool2") as scope:
        pool2=tf.nn.max_pool(conv4,[1,2,2,1],[1,2,2,1],padding="VALID",name='pooling2')

    with tf.variable_scope("conv5") as scope:
        weights = GenerateWegiths('weights',[3,3,64,128])
        conv = tf.nn.conv2d(pool2, weights, strides=[1, 1, 1, 1], padding='VALID')
        biases = GenerateBias('biases',[128])
        preActivition = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(preActivition, name=scope.name)

    with tf.variable_scope("conv6") as scope:
        weights = GenerateWegiths('weights',[3,3,128,128])
        conv = tf.nn.conv2d(conv5, weights, strides=[1, 1, 1, 1], padding='VALID')
        biases = GenerateBias('biases',[128])
        preActivition = tf.nn.bias_add(conv, biases)
        conv6 = tf.nn.relu(preActivition, name=scope.name)

    with tf.variable_scope("MaxPool3") as scope:
        pool3=tf.nn.max_pool(conv6,[1,2,2,1],[1,2,2,1],padding="VALID",name='pooling3')

    with tf.variable_scope("FC1") as scope:
        tShape=pool3.get_shape()
        fc1Shape=tShape[1].value*tShape[2].value*tShape[3].value
        reshape=tf.reshape(pool3,[-1,fc1Shape])
        fc1=tf.nn.dropout(reshape,keepProb,name='fc1Dropout')

    with tf.variable_scope('FC21') as scope:
        weights = GenerateWegiths('weights',[fc1Shape,10])
        biases = GenerateBias('biases',[10])
        fc21=tf.matmul(fc1,weights)+biases

    with tf.variable_scope("FC22") as scope:
        weights = GenerateWegiths('weights',[fc1Shape,10])
        biases = GenerateBias('biases',[10])
        fc22 = tf.matmul(fc1, weights) + biases

    #[batchsize,10]
    return fc21,fc22


def Loss(logits1,logits2,labels):
    #print(labels[:0])
    labels=tf.convert_to_tensor(labels,dtype=tf.int32)
    with tf.variable_scope("loss1") as scope:
        crossEntropy=tf.nn.sparse_softmax_cross_entropy_with_logits\
            (labels=labels[:,0],logits=logits1,name="entropyPerExample")
        loss1=tf.reduce_mean(crossEntropy,name="loss1")
        tf.summary.scalar(scope.name+'/loss1',loss1)

    with tf.variable_scope("loss2") as scope:
        crossEntropy=tf.nn.sparse_softmax_cross_entropy_with_logits\
            (labels=labels[:,1],logits=logits2,name="entropyPerExample")
        loss2=tf.reduce_mean(crossEntropy,name="loss2")
        tf.summary.scalar(scope.name+'/loss2',loss2)

    return loss1,loss2


def Training(loss1,loss2,learningRate):
    with tf.name_scope("optimizer1") as scope:
        optimizer1=tf.train.AdamOptimizer(learning_rate=learningRate)
        globalStep=tf.Variable(0,trainable=False,name="globalStep")
        trainOp1=optimizer1.minimize(loss1,global_step=globalStep)

    with tf.name_scope("optimizer2") as scope:
        optimizer2=tf.train.AdamOptimizer(learning_rate=learningRate)
        globalStep=tf.Variable(0,trainable=False,name="globalStep")
        trainOp2=optimizer2.minimize(loss2,global_step=globalStep)

    return trainOp1,trainOp2


def Evaluation(logits1,logits2,labels):
    #[batchSize,10(numClass)]
    #[3*batchSize,10(numClass)](第一幅图片三个输出位的输出结果，第二幅图三个输出位的输出结果。。。)
    logitsAll=tf.concat([logits1,logits2],0)
    #labels:稀疏的 [batchSize,3]
    #transpose:[3,batchSize]
    #reshape:[3*batchSize](每三个为图片三位的真是i标签)
    labels=tf.convert_to_tensor(labels,tf.int32)
    labelsAll=tf.reshape(tf.transpose(labels),[-1])
    with tf.variable_scope("accuracy") as scope:
        #print(sess.run(labelsAll))
        correct=tf.nn.in_top_k(logitsAll,labelsAll,1)
        correct=tf.cast(correct,tf.float16)
        accuracy=tf.reduce_mean(correct)
        tf.summary.scalar(scope.name+'/accuracy',accuracy)

    return accuracy


class ReadData:
    def __init__(self,imgPath):
        self.imgPath=imgPath
        self.imgList=os.listdir(imgPath)
        self.sign=0

    def NextBatch(self,batchNum):
        imgBatch=[];labels=[]
        indexL=self.sign
        indexU=self.sign+batchNum
        for index in range(indexL,indexU):
            imgName=self.imgList[index%len(self.imgList)]
            cLabels=list(map(int,imgName.split('_')[0]))
            labels.append(cLabels)
            absPath=os.path.join(self.imgPath,imgName)
            tempImg=cv2.imread(absPath)
            imgBatch.append(tempImg)
            self.sign=indexU%len(self.imgList)

        return np.array(labels),np.array(imgBatch)

if __name__=="__main__":
    path = '.\\temptrain'
    chekkPath = '.\\test3'
    testImgPath = '.\\yz'

    reader = ReadData(path)
    testreader = ReadData(chekkPath)

    print('start:', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    with tf.Session() as sess:
        imgsHolder = tf.placeholder(tf.float32, shape=[None, 54, 64, 3], name="inputs")
        labelsHolder = tf.placeholder(tf.int32, shape=[None, 3], name="inputLabels")
        keepHolder = tf.placeholder(tf.float32, name="keepPorb")

        logits1, logits2 = Inference(imgsHolder, keepHolder)
        loss1, loss2 = Loss(logits1, logits2, labelsHolder)
        trainOp1, trainOp2 = Training(loss1, loss2, 1e-4)
        accuracy = Evaluation(logits1, logits2, labelsHolder)

        inputImg = tf.summary.image('inputImg', imgsHolder, 10)
        merge = tf.summary.merge_all()

        trainWriter = tf.summary.FileWriter('./log2', sess.graph)
        saver = tf.train.Saver()

        initOp = tf.global_variables_initializer()
        sess.run(initOp)

        for i in range(40000):
            labels, imgs = reader.NextBatch(9)
            # print(imgs.shape)
            _, _, _, summary = sess.run([trainOp1, trainOp2, accuracy, merge],
                                        feed_dict={imgsHolder: imgs, labelsHolder: labels, keepHolder: 0.5})
            trainWriter.add_summary(summary, i)
            if i % 100 == 0:
                labels, imgs = testreader.NextBatch(29)
                acc = sess.run(accuracy, feed_dict={imgsHolder: imgs, labelsHolder: labels, keepHolder: 1})
                print("step %d with accuracy:" % i, acc)
            if (i + 1) % 500 == 0:
                saver.save(sess, './model2/model.ckpt', global_step=i + 1)

        trainWriter.close()