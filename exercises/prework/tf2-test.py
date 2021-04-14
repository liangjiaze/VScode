#导入所需要的依赖包  
import tensorflow as tf  
import numpy as np  

#实例化一个tf.keras.Sequential  
model=tf.keras.Sequential()  
#使用Sequential的add方法添加一层全连接神经网络 model.add(tf.keras.layers.Dense(input_dim=1,units=1))  

#使用Sequential的compile方法对神经网络模型进行编译，loss函数使用MSE，optimizer使用SGD（随机梯度下降）
model.compile(loss='mse',optimizer='sgd') 
#随机生成一些训练数据，在-10到10的范围内生成700个等差数列作为训练输入
X = np.linspace(-10, 10, 700) 
#通过一个简单的算法生成Y数据，模拟训练数据的标签 
Y=2*X+100+np.random.normal(0, 0.1, (700, )) 
#开始训练，“verbose=1”表示以进度条的形式显示训练信息，“epochs=200”表示训练的epochs为200，“validation_split=0.2”表示分离20%的数据作为验证数据
model.fit(X,Y,verbose=1,epochs=200,validation_split=0.2) 

#在完成神经网络模型的训练之后，可以使用Sequential的save方法将训练的神经网络模型保存为H5格式的模型文件
filename='line_model.h5'
model.save(filename) 
print("保存模型为line_model.h5") 

#加载神经网络模型需要使用tf.keras.models.load_model这个API，在完成模型的加载后可以使用Sequential的predict方法进行预测
x=tf.constant([0.5]) 
model=tf.keras.models.load_model(filename) 
y=model.predict(x) 
print(y)