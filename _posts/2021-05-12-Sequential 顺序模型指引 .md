---
published: false
---
# 1 、开始使用 Keras 顺序 (Sequential) 模型
顺序模型是多个网络层的线性堆叠。 你可以通过将层的列表传递给 Sequential 的构造函数，来创建一个 Sequential 模型：
```
import tensorflow as tff
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

model = Sequential([
   Dense(32, input_shape=(784,))
   Activation('relu')
   Dense(10)
   Activation ('softmax')])
```

也可以使用.add（）方法将各层添加到模型中:
```
model= Sequential O
model.add( Dense(32, input_dim=784))
model.add(Activation('relu'))
```

# 2 、指定输入数据的尺寸
模型需要知道它所期望的输入的尺寸。出于这个原因，顺序模型中的第一层（只有第一层， 因为下面的层可以自动地推断尺寸）需要接收关于其输入尺寸的信息。有几种方法来做到这一 点： 
• 
传递一个 input_shape 参数给第一层。它是一个表示尺寸的元组 (一个整数或 None 的元 组，其中 None 表示可能为任何正整数)。在input_shape 中不包含数据的 batch 大小。 
• 
某些 2D 层，例如 Dense，支持通过参数 input_dim 指定输入尺寸，某些 3D 时序层支持input_dim 和 input_length 参数。 
• 
如果你需要为你的输入指定一个固定的 batch 大小（这对 stateful RNNs 很有用），你可以传递一个 batch_size 参数给一个层。如果你同时将 batch_size=32 和 input_shape=(6, 8) 传递给一个层，那么每一批输入的尺寸就为 (32，6，8)。 
因此，下面的代码片段是等价的： 
```
model = Sequential()
model.add(Dense(32, input_shape=(784, )))

model = Sequential()
model.add(Dense(32, input_dim=784))
```
# 3 、编译
在训练模型之前，您需要配置学习过程，这是通过 compile 方法完成的。它接收三个参数： 
• 
优化器 optimizer。它可以是现有优化器的字符串标识符，如 rmsprop 或 adagrad，也可以是 Optimizer 类的实例。详见：optimizers。 
• 
损失函数 loss，模型试图最小化的目标函数。它可以是现有损失函数的字符串标识符，如categorical_crossentropy 或 mse，也可以是一个目标函数。详见：losses。 
• 
评估标准 metrics。对于任何分类问题，你都希望将其设置为 metrics = ['accuracy']。 
评估标准可以是现有的标准的字符串标识符，也可以是自定义的评估标准函数。
```
# 多分类问题
model.compile(optimizer='rmsprop',
         loss='categorical_crossentropy',
         metrics=['accuracy'])

# 二分类问题
model.compile(optimizer='rmsprop',
         loss='binary_crossentropy',
         metrics=['accuracy'])
         
# 均方误差回归问题
model.compile(optimizer='rmsprop',
         loss='mse')

# 自定义评估标准函数
import keras.backend as K

def mean_pred(y_true, y_pred):
   return K.mean(y_pred)
   
model.compile(optimizer=''rmsprop',
         loss='binary_crossentropy',
         metrics=['accuracy', mean_pred])
```
# 4 、训练
Keras 模型在输入数据和标签的 Numpy 矩阵上进行训练。为了训练一个模型，你通常会使 用 fit 函数。
***对于具有 2 个类的单输入模型（二进制分类）：***
```
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100)
model.add(Dense(1, activation=sigmoid'))
model.compile(optimizer='rmsprop',
         loss='binary_crossentropy',
         metrics=['accuracy'])

# 生成虚拟数据
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))

# 训练模型，以32个样本为一个batch进行迭代
model.fit(data, labels, epochs=10, batch_size=32)
```
***对于具有 10 个类的单输入模型（多分类分类）***
```
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
         loss='categorical_crossentropy',
         metrics=['accuracy'])

# 生成虚拟数据
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(10, size=(1000, 1))

# 将标签转换为分类的onne-hot编码
one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)
# 训练模型，以32个样本为一个batch进行迭代
model.fit(data, one_hot_labels, epochs=10, batch_size=32)
```
















