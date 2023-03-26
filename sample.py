import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense

# 准备数据
time_steps = 2

train_data = [1, 2, 3, 4, 5]
train_labels = [2, 4, 6, 8, 10]

# 创建模型
model = Sequential()
model.add(Dense(1, input_dim=1))

# 编译模型
model.compile(optimizer='adam', loss='mse')

# 训练模型
model.fit(train_data, train_labels, epochs=6000, batch_size=30)

# 预测新数据
x_test = [3, 6, 3, 3, 1]
prediction = model.predict(x_test)
print(prediction)

model.save('./model/sample/sample')
