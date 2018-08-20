# 基于深度学习的量体尺数字识别 #


## 1、说明 
+ roughdetect.py:基于opencv的cascade分类器，大体确定数字的位置
+ cache.py:缓存roughdetect检测出的subimg
+ crop.py:采用不同的阈值对图像进行二值化处理，然后进行轮廓检测，求包括轮廓的最小外接矩形。根据数字在图像中的占比及长宽比，对轮廓进行筛选，返回所有检测到的轮廓的左上角坐标和长宽
+ deskewH.py:根据crop返回的轮廓，求左上角和右下角坐标，然后做随即一致性检验，求出上边和下边的直线以及与图像边缘交点，crop出图像，进行仿射变换到标准大小
+ deskewV:遍历一个区间内的所有角度，调用crop.py求出包围轮廓的最最小矩形，求面积，最后求出竖直调整角度，进行仿射变
+ model.py:定义了图片数字的预测模型，采用e2e(端到端)方式进行预测
+ predict.py:加载模型并进行预测
+ ckpt2pb.py:将tensorflow保存的ckpt模型转换为pb模型
+ pipline.py:最终运行文件
+ cache:cache.py缓存的subimg
+ log:tensorboard日志文件
+ model:model.py训练保存的模型参数

## 2、运行方式 ##
python pipline.py --imagepath='.\\testimgs\\1.jpg' --cached=True

## 3、参考 ##
[Relocy的专栏](https://blog.csdn.net/relocy)

[github](https://github.com/zeusees/HyperLPR)