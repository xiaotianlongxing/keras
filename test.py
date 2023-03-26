import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.models import Sequential

# 数据准备
time_steps = 10
batch_size = 64
train_size = 1000
test_size = 100
train_X = np.random.rand(train_size, time_steps, 1)
train_y = np.random.rand(train_size, 1)
test_X = np.random.rand(test_size, time_steps, 1)
test_y = np.random.rand(test_size, 1)

# 搭建卷积神经网络模型
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(time_steps, 1)))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(1))

# 编译模型
model.compile(optimizer='adam', loss='mse')

# 训练模型
model.fit(train_X, train_y, epochs=10, batch_size=batch_size, verbose=1)

# 评估模型
mse = model.evaluate(test_X, test_y, verbose=1)
print("Mean Squared Error:", mse)

# 进行预测
pred_y = model.predict(test_X)
print("Predictions:", pred_y)


#该示例代码使用了一个包含两个卷积层、一个最大池化层、一个全连接层和一个输出层的卷积神经网络模型。训练数据和测试数据均为时间序列数据，其中train_X和test_X的shape为(batch_size, time_steps, 1)，train_y和test_y的shape为(batch_size, 1)。模型使用均方误差（MSE）作为损失函数，优化器选择Adam。训练完成后，评估模型性能并使用模型进行预测。