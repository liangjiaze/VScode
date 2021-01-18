# 引入要用的包
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 用pandas读取csv数据文件
data = pd.read_csv('E:/bigdata/data/tensorflow-data/Advertising.csv')
# 显示部分数据及格式
# data.head

# 使用pyplot查看数据的分布，看相关关系
# plt.scatter(data.TV,data.sales)
# plt.scatter(data.radio,data.sales)
# plt.scatter(data.newspaper,data.sales)

# 定义x为所有行的第二至倒数第二列
x = data.iloc[:, 1:-1]

# 定义y为所有行的倒数第一列
y = data.iloc[:, -1]

# 使用Sequential模型
# 定义了
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10,input_shape=(3,), activation='relu'),
    tf.keras.layers.Dense(1)
])

# 显示模型的所有参数
model.summary()

# 配置训练方法时，告知训练时用的优化器、损失函数和准确率评测标准
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=None
)

# 使用model.fit()方法来执行训练过程
# 告知训练集的输入以及标签
# 迭代次数epochs为500
model.fit(x, y, epochs=1300)

# 定义测试数据
test = data.iloc[:10, 1:-1 ]

# 使用模型预测测试数据
model.predict(test)

# 定义原始数据并展示
# test1 = data.iloc[:10, -1]
# test1