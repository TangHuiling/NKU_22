import tensorflow as tf
gpus = tf.config.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)

import keras
#ImageDataGenerator()是keras.preprocessing.image模块中的图片生成器，
# 同时也可以在batch中对数据进行增强，扩充数据集大小，增强模型的泛化能力。比如进行旋转，变形，归一化等等。
from  keras.preprocessing.image import ImageDataGenerator
# 序贯模型是函数式模型的简略版，为最简单的线性、从头到尾的结构顺序，不分叉，是多个网络层的线性堆叠。
# Keras实现了很多层，包括core核心层，Convolution卷积层、Pooling池化层等非常丰富有趣的网络结构。
# 我们可以通过将层的列表传递给Sequential的构造函数，来创建一个Sequential模型。
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D
import os

num_classes=6#训练模型时要处理的类即情感的种类数。
img_rows,img_cols=48,48#馈送到神经网络中的图像阵列大小。
batch_size=32#更新模型之前处理的样本数量。epochs 是完整通过训练数据集的次数。batch_size必须大于等于1并且小于或等于训练数据集中的样本数。

# 现在让我们开始加载模型，这里使用的数据集是fer2013，该数据集是由kaggle托管的开源数据集。
# 数据集共包含7类，分别是愤怒、厌恶、恐惧、快乐、悲伤、惊奇、无表情，训练集共有28,709个示例。
# 我们将数据储存在特定文件夹中。例如，“愤怒”文件夹包含带有愤怒面孔等的图片。
# 在这里，我们使用5类，包括“愤怒”，“快乐”，“悲伤”，“惊奇”和“无表情”。
# 使用24256张图像作为训练数据，3006张图像作为检测数据。
train_data_dir='fer2013/train'
validation_data_dir='fer2013/validation'

# 导入了检测和训练数据。该模型是在训练数据集上进行训练的；
# 在检测数据集上检测该模型性能，检测数据集是原始数据集的一部分，从原始数据集上分离开来的

#对这些数据集进行图像增强。图像数据增强可以扩展训练数据集大小，改善图像质量。
# Keras深度学习神经网络库中的ImageDataGenerator类通过图像增强来拟合模型
train_datagen = ImageDataGenerator(
    #rescale: 重缩放因子。默认为 None。如果是 None 或 0，不进行缩放，否则将数据乘以所提供的值（在应用任何其他转换之前）。
    #rescale的作用是对图片的每个像素值均乘上这个放缩因子，这个操作在所有其它变换操作之前执行，
    # 在一些模型当中，直接输入原图的像素值可能会落入激活函数的“死亡区”，
    # 因此设置放缩因子为1/255，把像素值放缩到0和1之间有利于模型的收敛，避免神经元“死亡”。
    #输入进行标准化，由于图像的像素值将在0到1之间，对输入进行归一化的原因与数值稳定性和收敛性有关
    #神经网络收敛的机会更高，并且梯度下降/ adam算法更有可能保持稳定
    rescale=1./255,#图像预处理，归一化
    rotation_range=30,#整数，随机旋转，在这里我们使用30度。

    #所谓shear_range就是错切变换，效果就是让所有点的x坐标(或者y坐标)保持不变，而对应的y坐标(或者x坐标)则按比例发生平移，
    # 且平移的大小和该点到x轴(或y轴)的垂直距离成正比。
    shear_range=0.3,#剪切强度（逆时针方向的剪切角，以度为单位）。在这里我们使用0.3作为剪切范围。

    #浮点数 或 [lower, upper]。随机缩放范围。如果是浮点数，[lower, upper] = [1-zoom_range, 1+zoom_range]。
    # zoom_range参数可以让图片在长或宽的方向进行放大，可以理解为某方向的resize，因此这个参数可以是一个数或者是一个list。
    # 当给出一个数时，图片同时在长宽两个方向进行同等程度的放缩操作；当给出一个list时，则代表[width_zoom_range, height_zoom_range]，
    # 即分别对长宽进行不同程度的放缩。而参数大于0小于1时，执行的是放大操作，当参数大于1时，执行的是缩小操作。
    zoom_range=0.3,#随机缩放的范围，这里我们使用0.3作为缩放范围。

    width_shift_range=0.4,#在图像的整个宽度上移动一个值。
    height_shift_range=0.4,#这会在整个图像高度上移动一个值。
    horizontal_flip=True,#水平翻转图像。
    fill_mode='nearest')#通过上述使用的方法更改图像的方向后填充像素，使用“最近”作为填充模式，即用附近的像素填充图像中丢失的像素。

#在这里，我只是重新保存验证数据，而没有执行任何其他扩充操作，因为我想使用与训练模型中数据不同的原始数据来检查模型。
validation_datagen = ImageDataGenerator(rescale=1./255)


# 获取图像路径，生成批量增强数据。该方法只需指定数据所在的路径，而无需输入numpy形式的数据，也无需输入标签值，
# 会自动返回对应的标签值。返回一个生成(x, y)元组的DirectoryIterator。

train_generator = train_datagen.flow_from_directory(
 train_data_dir,#目标目录的路径。每个类应该包含一个子目录。任何在子目录树下的 PNG, JPG, BMP, PPM 或 TIF 图像，都将被包含在生成器中。
 color_mode='grayscale',# "grayscale", "rbg" 之一。默认："rgb"。图像是否被转换成 1 或 3 个颜色通道。
 target_size=(img_rows,img_cols),#整数元组 (height, width)，默认：(256, 256)。所有的图像将被调整到的尺寸。
 batch_size=batch_size,#一批数据的大小（默认 32）。

 #"categorical", "binary", "sparse", "input" 或 None 之一。
 # 默认："categorical"。决定返回的标签数组的类型：
 class_mode='categorical',#"categorical" 将是 2D one-hot 编码标签，
 shuffle=True)#是否混洗数据（默认 True）。

validation_generator = validation_datagen.flow_from_directory(
 validation_data_dir,
 color_mode='grayscale',
 target_size=(img_rows,img_cols),
 batch_size=batch_size,
 class_mode='categorical',
 shuffle=True)

#在上面的代码中，我正在使用flow_from_directory（）方法从目录中加载我们的数据集，
# 该目录已扩充并存储在train_generator和validation_generator变量中。flow_from_directory（）采用目录的路径并生成一批扩充数据。
# 因此，在这里，我们为该方法提供了一些选项，以自动更改尺寸并将其划分为类，以便更轻松地输入模型。

#
#
# # directory：数据集的目录。
#
# • color_mode：在这里，我将图像转换为灰度，因为我对图像的颜色不感兴趣，而仅对表达式感兴趣。
#
# • target_size：将图像转换为统一大小。
#
# • batch_size：制作大量数据以进行训练。
#
# • class_mode：在这里，我将“类别”用作类模式，因为我将图像分为6类。
#
# • shuffle：随机播放数据集以进行更好的训练。

#以上为数据集的预处理

#构建CNN网络，模型的大脑，使用的是Sequential模型，该模型定义网络中的所有层将依次相继并将其存储在变量模型中。

model=Sequential()

# 网络由七个块组成

#Con2D,二维卷积层，该层对二维输入进行滑动窗卷积，当使用该层作为第一层时，应提供input_shape参数
#32-卷积核的数目，输出的维度
#（3，3）-卷积核的宽度和长度
#步长默认为1
#padding：补0策略，为“valid”, “same” 。“valid”代表只进行有效的卷积，
# 即对边界数据不处理。“same”代表保留边界处的卷积结果，通常会导致输出shape与输入shape相同。
#keras不同的层可能使用不同的关键字来传递初始化方法，一般来说指定初始化方法的关键字是kernel_initializer 和 bias_initializer
#kernel_initializer=‘he_normal'——He正态分布初始化方法，参数由0均值，标准差为sqrt(2 / fan_in) 的正态分布产生，其中fan_in权重张量的扇入
#input_shape代表48*48的灰度图像
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',
                 input_shape=(img_rows,img_cols,1)))

#融合了sigmoid和ReLU，左侧具有软饱和性，右侧无饱和性。
#右侧线性部分使得ELU能够缓解梯度消失，而左侧软饱能够让ELU对输入变化或噪声更鲁棒。
#ELU的输出均值接近于零，所以收敛速度更快。
#elu和relu的区别在负区间，relu输出为0，而elu输出会逐渐接近-α，更具鲁棒性。elu激活函数另一优点是它将输出值的均值控制为0
model.add(Activation('elu'))

#批标准化, 和普通的数据标准化类似, 是将分散的数据统一的一种做法, 也是优化神经网络的一种方法.
#具归一化每一层的激活，即将平均激活值保持在接近0并将激活标准偏差保持在接近1。
model.add(BatchNormalization())


model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',
                 input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())

# 此层为网络创建卷积层。我们创建的该层包含32个大小为（3,3）滤波器，
# 其中使用padding ='same'填充图像并使用内核初始化程序he_normal。
# 添加了2个卷积层，每个层都有一个激活层和批处理归一化层。

#通过沿pool_size定义的沿特征轴的每个尺寸的窗口上的最大值，对输入表示进行下采样。在此， pool_size大小为（2,2）
#通过为输入的每个通道在输入窗口(大小由 pool_size 定义)上取最大值，沿其空间维度(高度和宽度)对输入进行下采样。
model.add(MaxPooling2D(pool_size=(2,2)))#整数或长为2的整数tuple，代表在两个方向（竖直，水平）上的下采样因子，如取（2，2）将使图片在两个维度上均变为原长的一半。
#为输入数据施加Dropout。Dropout将在训练过程中每次更新参数时按一定概率（rate）随机断开输入神经元，Dropout层用于防止过拟合。
model.add(Dropout(0.2))
#Block-2
#与block-1相同的层，但是卷积层具有64个滤波器。
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
#Block-3
# 与block-1相同的层，但是卷积层具有128个滤波器
model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
#Block-4
# 与block-1相同的层，但是卷积层具有256个滤波器。
model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#Block-5
#Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。
#将前一层的输出展平，即转换为矢量形式。
model.add(Flatten())
#密集层-该层中每个神经元都与其他每个神经元相连。在这里，我使用带有内核的程序初始化64个单元或64个神经元-he_normal。
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
#这些层之后使用elu激活，批处理归一化，最后以dropout为50％选择忽略
#Block-6
#与模块5相同的层，但没有展平层，因为该模块的输入已展平
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
#Block-7
#密集层-网络的最后一个块中，我使用num_classes创建一个密集层，该层具有he_normal初始值设定项，其unit =类数。
# • 激活层-在这里，我使用softmax，该层多用于分类。
model.add(Dense(num_classes,kernel_initializer='he_normal'))
model.add(Activation('softmax'))

#检查模型的整体结构：
#这是一个大型网络，包含1,328,037个 参数。
print(model.summary())

#编译和训练

#机器学习包括两部分内容，一部分是如何构建模型，另一部分就是如何训练模型。训练模型就是通过挑选最佳的优化器去训练出最优的模型
#在编译一个Keras模型时，优化器是2个参数之一（另外一个是损失函数）
#优化器就是向模型打包传递参数，什么参数呢，就是我们训练时使用到的诸如，学习率，衰减，momentum，梯度下降得到若干种方式，用不用动量等等。
#你可以在一开始传入这个值，然后就一直使用这个值训练，也可以在训练时根据epoch调整参数。

from keras.optimizers import RMSprop,SGD,Adam



#回调函数是一组在训练的特定阶段被调用的函数集，你可以使用回调函数来观察训练过程中网络内部的状态和统计信息。
# 通过传递回调函数列表到模型的.fit()中，即可在给定的训练阶段调用该函数集中的函数。

# callbacks来控制正在训练的模型
# 最开始训练过程是先训练一遍，然后得到一个验证集的正确率变化趋势，从而知道最佳的epoch，设置最佳epoch，再训练一遍得到最终结果，十分浪费时间！！！
#
# 节省时间的一个办法是在验证集准确率不再上升的时候，终止训练。keras中的callback可以帮我们做到这一点。
# callback是一个obj类型的，他可以让模型去拟合，也常在各个点被调用。它存储模型的状态，能够采取措施打断训练，保存模型，加载不同的权重，或者替代模型状态。
# callbacks可以用来做这些事情：
#
# 模型断点续训：保存当前模型的所有权重
# 提早结束：当模型的损失不再下降的时候就终止训练，当然，会保存最优的模型。
# 动态调整训练时的参数，比如优化的学习率。

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

#它将监视验证损失，并使用mode ='min'属性尝试将损失降至最低。
#到达检查点时，它将保存训练有素的最佳大小。Verbose = 1仅用于代码创建检查点时的可视化。
#该回调函数将在每个epoch后保存模型到filepath
checkpoint = ModelCheckpoint('EmotionDetectionModel.h5',#保存模型文件的路径，这里我保存的模型文件名为EmotionDetectionModel.h5
                             monitor='val_loss',#要监视的值。在这里，我正在监视验证损失。
                             #‘auto’，‘min’，‘max’之一，在save_best_only=True时决定性能最佳模型的评判准则，
                             # 当监测值为val_acc时，模式应为max，当检测值为val_loss时，模式应为min。在auto模式下，评价准则由被监测值的名字自动推断。
                             mode='min',
                             save_best_only=True,#当设置为True时，将只保存在验证集上性能最好的模型
                             verbose=1)#信息展示模式
                                       # verbose = 0 为不在标准输出流输出日志信息
                                       # verbose = 1 为输出进度条记录
                                       # verbose = 2 为每个epoch输出一行记录
                                       # 默认为 1


#通过检查以下属性，以提前结束运行。
earlystop = EarlyStopping(monitor='val_loss',#要监视的数量。在这里，我正在监视验证损失。
                          min_delta=0,#被监视的数量的最小变化有资格作为改进，即绝对变化小于min_delta将被视为没有任何改进。在这里我给了0。
                          patience=3,#没有改善的时期数，此后将停止训练。我在这里给了它3。
                          verbose=1,#为输出进度条记录
                          restore_best_weights=True
                          )

#当评价指标不在提升时，减少学习率
#当学习停滞时，减少2倍或10倍的学习率常常能获得较好的效果。
#该回调函数检测指标的情况，如果在patience个epoch中看不到模型性能提升，则减少学习率
reduce_lr = ReduceLROnPlateau(monitor='val_loss',#要监视的数量。在这里，我正在监视验证损失。
                              factor=0.2,#每次减少学习率的因子，学习率将以lr = lr*factor的形式被减少
                              patience=3,#当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
                              verbose=1,
                              min_delta=0.0001)#在被监测的数据中被认为是提升的最小变化， 例如，小于 min_delta 的绝对变化会被认为没有提升。


callbacks = [earlystop,checkpoint,reduce_lr]
#最后使用编译模型model.compile（）和适合训练数据集的模型model.fit_generator（）


# 如何选择优化器
# 如果数据是稀疏的，就用自适用方法，即 Adagrad, Adadelta, RMSprop, Adam。
#
# RMSprop, Adadelta, Adam 在很多情况下的效果是相似的。
#
# Adam 就是在 RMSprop 的基础上加了 bias-correction 和 momentum，
#
# 随着梯度变的稀疏，Adam 比 RMSprop 效果会好。
#
# 整体来讲，Adam 是最好的选择。
#
# 很多论文里都会用 SGD，没有 momentum 等。SGD 虽然能达到极小值，但是比其它算法用的时间长，而且可能会被困在鞍点。
#
# 如果需要更快的收敛，或者是训练更深更复杂的神经网络，需要用一种自适应的算法。

model.compile(loss='categorical_crossentropy',#此值将确定要在代码中使用的损失函数的类型。
              # 在这里，我们有6个类别或类别的分类数据，因此使用了“ categorical_crossentropy”损失
              optimizer = Adam(lr=0.001),#此值将确定要在代码中使用的优化器功能的类型。
              # 这里我使用的学习率是0.001的Adam优化器，因为它是分类数据的最佳优化器。
              metrics=['accuracy'])#metrics参数应该是一个列表，模型可以有任意数量的metrics。
              # 它是模型在训练和测试过程中要评估的metrics列表。这里我们使用了精度作为度量标准。

nb_train_samples =28353
nb_validation_samples = 3534
epochs=20


#利用Python的生成器，逐个生成数据的batch并进行训练。生成器与模型将并行执行以提高效率。
# 例如，该函数允许我们在CPU上进行实时的数据提升，同时在GPU上进行模型训练

history=model.fit_generator(
                train_generator,#我们之前创建的train_generator对象。
                steps_per_epoch=nb_train_samples//batch_size,#当生成器返回steps_per_epoch次数据时计一个epoch结束，执行下一个epoch
                epochs=epochs,#数据迭代的轮数
                callbacks=callbacks,#包含我们之前创建的所有回调的列表。
                validation_data=validation_generator,#我们之前创建的validation_generator对象。
                validation_steps=nb_validation_samples//batch_size)#当validation_data为生成器时，本参数指定验证集的生成器返回次数
