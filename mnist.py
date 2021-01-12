import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pandas as pd

# 读取MNIST数据。
#分别读入MNIST数据集的训练集数据和测试集数据
(X_train_image, y_train_label), (X_test_image, y_test_label) = tf.keras.datasets.mnist.load_data()

# 打印MNIST数据的尺寸大小
print(len(X_train_image))
print(len(X_test_image))
print(X_train_image.shape)

# 显示一张图片
def plot_image(image):
    fig=plt.gcf()    #图表生成
    fig.set_size_inches(3,3)  #设置图表大小
    plt.imshow(image,cmap='binary') #以黑白灰度显示图片
    plt.show()    #开始绘图
plot_image(X_train_image[0]) #显示一张图片

# 为了更加方便查看图像、标签及预测结果之间对应的关系，定义一个函数来显示。
def plot_image_labels_prediction(images,labels,prediction,idx,nums=10):
    fig = plt.gcf() 
    fig.set_size_inches(12,14)  #设置图表大小
    if nums>25: nums=25 #最多显示25张图像
    for i in range(0,nums):
        ax = plt.subplot(5,5,1+i) #子图生成
        ax.imshow(images[idx],cmap='binary') #idx是为了方便索引所要查询的图像
        title = 'label=' + str(labels[idx]) #定义title方便图像结果对应
        if(len(prediction)>0): #如果有预测图像，则显示预测结果
            title += 'prediction='+ str(prediction[idx])
        ax.set_title(title,fontsize=10) #设置图像title
        ax.set_xticks([]) #无x刻度
        ax.set_yticks([]) #无y刻度
        idx+=1
    plt.show()
plot_image_labels_prediction(X_train_image,y_train_label,[],0,10) #显示前10张的图像


# 将二维image转化为一维向量, 转化过程是图像大小相乘，结果为：28x28 = 784, 以便于能作为神经网络的第一层输入。
X_train = X_train_image.reshape(60000,784).astype('float32') 
X_test = X_test_image.reshape(10000,784).astype('float32')

# 对图像数据进行归一化。归一化是必要操作，可以提高准确率。原因是如果没有归一化那么图像的数据值是0-255，不同的图像之间的值差异很大，在训练过程中，小的值很容易被忽略。因此我们需要对数据进行缩放，常见的一种方法是将数据的值压缩到(0,1)区间内，在本例中，直接将数据除以255即可。
X_train_normalize = X_train/255 
X_test_normalize = X_test/255

# 对数据标签进行One-hot编码（独热编码），我们都知道数字识别的结果是0-9，但为了满足神经网络的最后一层的10个神经元输出，我们需要将数字转换为二进制的10个0和1的组合，具体可看下面代码例子，第一个为数据的原标签，第二个是One-hot编码后的标签。
y_train_label[:5] #显示前5个图像的对应标签数字

# 调用keras的utils中的to_categorical函数进行直接数字编码。
y_TrainOneHot = tf.keras.utils.to_categorical(y_train_label)
y_testOneHot = tf.keras.utils.to_categorical(y_test_label)
y_TrainOneHot[0:5] #显示前5个数据编码后的结果

# 建立线性堆叠模型Sequential，使用它我们可以搭建出很多复杂的神经网络模型，后面的层就犹如叠面包一样一层一层通过add指令叠加上去就好了，十分方便
model = tf.keras.models.Sequential() # 调用Sequential模型

# 不断叠加层上去，Dense层是全连接层，指的是神经前后前后两层神经元中两两都有互相连接，而Dropout的层的作用是为了防止过拟合（过拟合概念可自行查询），随机使多少比例的神经元失活，等于是断了之间的连接关系。
#units是该层神经元数量，input_dim是数据输入大小，kernel_initializer=normal指的是用正态分布来初始化神经元的权重和偏差，activation指的是激活函数
model.add(layers.Dense(units=256,input_dim=784,kernel_initializer='normal',activation='relu')) 
model.add(layers.Dropout(0.5)) #使50%的神经元失活
#最终输出是0-9，因此是10个神经元，softmax激活函数可以将输出转化为预测每个数字的概率
model.add(layers.Dense(units=10,kernel_initializer='normal',activation='softmax'))


#显示模型信息，其中Param指的是该层神经元总的参数数量，具体的计算公式是 Param=（上一层神经元数量）x（本层神经元数量）+（本层神经元数量）
print(model.summary())

#loss这里采用的交叉熵损失，优化器采用的是adam优化器，评估方式为准确率
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# 开始训练，其中validation_split是将原数据集划分一部分为验证集，epoch是训练次数，batch_size是每一批训练的数据量，verbose=2代表显示所有训练过程。
train_history = model.fit(x=X_train_normalize,y = y_TrainOneHot,validation_split=0.2,epochs=10,batch_size=200,verbose=2)

# 可以通过train_history查看训练过程，之前的训练步骤的值都保存在这里面。这里共有loss,accuracy,val_loss,val_accuracy四个参数。
train_history.history

# 可以定义函数，显示训练的曲线图。
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.xlabel('epoch')
    plt.ylabel(train)
    plt.legend(['train','validation'],loc='upper left')
show_train_history(train_history,'accuracy','val_accuracy')

# 通过evaluate可以使用模型在测试集上进行测试。
scores = model.evaluate(X_test_normalize,y_testOneHot)
print('accuracy=',scores[1])

# 对模型进行预测，并使用之前我们定义的函数plot_image_labels_prediction，可以方便查看预测结果之间的对应关系。
prediction = model.predict_classes(X_test)
plot_image_labels_prediction(X_test_image, y_test_label, prediction, idx=340)

# 通过pandas中的crosstab调用混淆矩阵，来查看哪些结果最容易混淆，对角线部分，为正确预测的部分，其它部分则是预测错误的部分
pd.crosstab(y_test_label,prediction,rownames=['label'])

# 查看是5预测错成3的数据位置
df = pd.DataFrame({'label':y_test_label,'predict':prediction})
print(df[(df.label==5)&(df.predict==3)])

# 通过我们先前定义的plot_image_labels_prediction函数，也可以直接查看预测错误的图像。
plot_image_labels_prediction(X_test_image, y_test_label, prediction, idx=340, nums=1)