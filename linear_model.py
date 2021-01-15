import tensorflow as tf
import pandas as pd
data = pd.read_csv('E:/bigdata/data/tg_ppq&upq.csv')

data

import matplotlib.pyplot as plt
plt.scatter(data.PPQ-data.UPQ,data.LINELOSS_RATE)


x = data.PPQ-data.UPQ
y = data.LINELOSS_RATE

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape=(1,)))

# 显示模型的参数信息
model.summary()

model.compile(optimizer='adam',loss='mse',)

history = model.fit(x,y,epochs=20)

# 用训练好的模型预测
# model.predict(x)
model.predict(pd.Series([68]))