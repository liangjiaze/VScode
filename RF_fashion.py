import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

# 传统机器学习模型预测fashion数据。

# 1.处理成二维数据
# 将原来的600002828的数据集，转换为60000*784的二维矩阵。

(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

train_images2=train_images.reshape(train_images.shape[0],train_images.shape[1]*train_images.shape[2])
train_images2.shape


# 由于将28*28图片转换成数组，特征输入相当于是784纬，尝试进行模型训练，在我的小笔记上完全跑不出来，没办法，这里只取了前1000张图片进行训练。

train_images2=train_images2[0:1000]
train_labels=train_labels[0:1000]

# 2.构造随机森林模型

# 构建模型代码
def model_RandomForest(Xtrain,Ytrain,is_optimize=1):
    """
    训练随机森林
    """
    RandomForest = RandomForestClassifier()
    if is_optimize==1:
        param_grid_Log_RegRandomForest = [{"n_estimators":list(range(50,100,10)),"max_depth":list(range(3,10)),"min_samples_split":list(range(100,500,50))}]
        score = make_scorer(accuracy_score)
        RandomForest = GridSearchCV(RandomForest,param_grid_Log_RegRandomForest,score,cv=3)
    RandomForest = RandomForest.fit(train_images2, train_labels)
    return RandomForest
def trainmodel_RF(x_train_stand,y_train):
    """
    目的：训练模型，随机森林
    x_train_stand：训练集输入，
    y_train：训练集标签，与输入相对应
    """
    #训练模型
    print("随机森林训练中")
    RandomForest = model_RandomForest(trainmodel_RF, train_labels)

    with open("RandomForestModel50.pickle","wb") as pickle_file:
        pickle.dump(RandomForest,pickle_file)   #随机森林的拟合模型

    Str = "随机森林训练完成"
    return print(Str)

# 开始执行训练，1000张图片大概10分钟左右，比TensorFlow慢多了（5次迭代1分钟左右，准确率达0.9），如果训练原始60000张图片，时间上大家可以想象一下。

trainmodel_RF(train_images2,train_labels)

# 3.验证随机森林模型效果

# 首先，转换测试集为矩阵。
test_images2=test_images.reshape(test_images.shape[0],test_images.shape[1]*test_images.shape[2])
test_images2.shape

# 载入模型

with open("RandomForestModel50.pickle","rb") as pickle_file:
    RandomForest = pickle.load(pickle_file)   #随机森林的拟合模型

# 预测结果，这里要用predict_proba方法。
#RandomForest.predict(test_images2)
RandomForest.predict_proba(test_images2)[0]

# 打印准确率，准确率0.742（已经很不错了，我们只是用了区区1000张图片进行训练，TensorFlow可是用了60000张图片）
print("随机森林准确率: {:.3f}".format(accuracy_score(test_labels,RandomForest.predict(test_images2))))

# 可视化结果
predictions_pro=RandomForest.predict_proba(test_images2)
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions_pro, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions_pro, test_labels)
plt.show()