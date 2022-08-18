#拿到数据集之后需要清洗数据，因此要import pandas
import pandas as pd
#引入numpy，要转成keras能读入的一个数组
import numpy as np
#构建CNN，要import keras相关

from keras.models import Sequential
from keras.layers import Conv2D,Dense,Dropout,Flatten,MaxPooling2D
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam
from keras.utils import np_utils

#先读入csv的一个数据集
Catch_bat=pd.read_csv('fer2013.csv')
#emotion-表情 pixels-像素 usage-标签向量
#print(Catch_bat.info())
#print(Catch_bat['Usage'].value_counts())
#print(Catch_bat.head())

#X_train-像素 Y_train-标签
X_train,Y_train,X_test,Y_test=[],[],[],[]

for index,row in Catch_bat.iterrows():
   # print(index)
   # print(row)
   val=row['pixels'].split(' ')
   #print(val)
   try:
       if 'Training' in row['Usage']:
           X_train.append(np.array(val,'float32'))
           Y_train.append(row['emotion'])
       elif 'PublicTest' in row['Usage']:
           X_test.append(np.array(val,'float32'))
           Y_test.append(row['emotion'])
   except:
       pass

#出错原因！！！：缩进真的很重要，这个print一定要顶格写，否则会被当作上面的循环里面的语句被一直循环

# print(f'X_train:{X_train[0:2]}')
# print(f'Y_train:{Y_train[0:2]}')
# print(f'X_test:{X_test[0:2]}')
# print(f'Y_test:{Y_test[0:2]}')

#转化数据到keras
X_train=np.array(X_train,'float32')
Y_train=np.array(Y_train,'float32')
X_test=np.array(X_test,'float32')
Y_test=np.array(Y_test,'float32')
print(X_train)
print(Y_train)

#进行数据标准化处理（进行均值处理和标准差的矫正）
X_train-=np.mean(X_train,axis=0)#计算每一列的平均均值
X_train/=np.std(X_train,axis=0)#计算每一列的标准差

X_test-=np.mean(X_test,axis=0)#计算每一列的平均均值
X_test/=np.std(X_test,axis=0)#计算每一列的标准差

#将标签转化为onehot编码，num_classes代表多少个分类

labels=7
Y_train=np_utils.to_categorical(Y_train,num_classes=labels)
Y_test=np_utils.to_categorical(Y_test,num_classes=labels)



features=64#卷积输出的滤波器的数量
batch_size=64
epochs=1
width,height=48,48

#生成新列表，修改数据

X_train=X_train.reshape(X_train.shape[0],width,height,1)#1代表一个特征，一种情绪
X_test=X_test.reshape(X_test.shape[0],width,height,1)

#构建CNN网络

model=Sequential()

#第一部分

#卷积取的局部特征，卷积层的作用是通过不断改变卷积核，来确定图片的特征有用的是哪些，得到再与相应的卷积核相乘输出矩阵
#进行卷积操作
#出错原因，X_tarin要变为X_train.shape
model.add(Conv2D(features,kernel_size=(3,3),activation='relu',input_shape=(X_train.shape[1:])))
model.add(Conv2D(features,kernel_size=(3,3),activation='relu'))


#进行池化层操作，保留图片最重要的特征

model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

#让50%神经元不工作

model.add(Dropout(0.5))

#第二部分
model.add(Conv2D(features,kernel_size=(3,3),activation='relu'))
model.add(Conv2D(features,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.5))

#第三部分
model.add(Conv2D(2*features,kernel_size=(3,3),activation='relu'))
model.add(Conv2D(2*features,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

#卷积层过渡到全连接层
model.add(Flatten())

#1024个神经元，把卷积提取的特征组装成一张图片，所以叫全连接层，实现分类
model.add(Dense(2*2*2*2*features,activation='relu'))

#防止过拟合，让20%的神经元不工作

model.add(Dropout(0.2))

#输出7个结果
model.add(Dense(labels,activation='softmax'))

#loss_function 交叉熵损失函数，计算准确率，指定优化器
model.summary()
model.compile(loss=CategoricalCrossentropy(),optimizer=Adam(),metrics=['accuracy'])

#训练模型,显示训练的进度条,validation_data：根据数据集上的效果进行调整，shuffle进行随机打乱
model.fit(X_train,Y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_test,Y_test),shuffle=True)

#保存模型
#保存网络结构
fer_json=model.to_json()
with open('fer.json','w') as json_file:
    json_file.write(fer_json)



#保存模型参数

model.save_weights('fer.h5')