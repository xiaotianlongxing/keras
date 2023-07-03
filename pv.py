import numpy as np
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
from sklearn.preprocessing import MinMaxScaler

# 假设这是最近一个月每天的网站PV数据，可以根据实际情况进行替换
data = [100, 120, 115, 150, 130, 140, 135, 160, 155, 170, 165, 180, 175, 190, 185, 200, 195, 210, 205, 220, 215, 230, 225, 240, 235, 250, 245, 260, 255, 270]

# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(np.array(data).reshape(-1, 1))

# 构建训练数据集
X_train, y_train = [], []
timesteps = 5  # 时间步长，用过去5天的PV数据预测当天的PV
for i in range(len(data)-timesteps):
    X_train.append(data[i:i+timesteps, 0])
    y_train.append(data[i+timesteps, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# 调整输入数据的形状（样本数，时间步长，特征数）
X_train = np.reshape(X_train, (X_train.shape[0], timesteps, 1))

# 构建RNN模型
model = Sequential()
model.add(SimpleRNN(units=50, input_shape=(timesteps, 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
model.fit(X_train, y_train, epochs=100, batch_size=32)

# 使用模型进行预测
last_month_data = data[-timesteps:].reshape(1, timesteps, 1)
print(last_month_data)
predicted_value = model.predict(last_month_data)
predicted_value = scaler.inverse_transform(predicted_value)

model.save('./model/serving/1')

print("明日的PV预测值为:", predicted_value)


input_data = {"instances": last_month_data.tolist()}


# 发送预测请求
import requests
# 发送POST请求到TensorFlow Serving服务器
response = requests.post('http://www.aiyuezhicheng.com:8501/v1/models/my_model:predict', json=input_data)
print(response.json())
predictions = response.json()['predictions']
predicted_value = scaler.inverse_transform(predictions)[0][0]

print("使用tf-serving的明日的PV预测值为:", predicted_value)