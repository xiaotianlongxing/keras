import numpy as np
from PIL import Image
from tensorflow import keras


# 飞机（airplane）
# 汽车（automobile）
# 鸟（bird）
# 猫（cat）
# 鹿（deer）
# 狗（dog）
# 青蛙（frog）
# 马（horse）
# 船（ship）
# 卡车（truck）

# 加载图像
img = Image.open('D:\workspace\python\c2.jpg')
img = img.resize((32, 32))  # 将图像大小调整为模型输入大小
img_array = np.array(img)
img_array = img_array.astype('float32') / 255.0  # 数据预处理

# 将图像输入模型进行预测
model = keras.models.load_model('./model/cifar10/cifar10')
result = model.predict(np.array([img_array]))
predicted_class = np.argmax(result[0])

# 输出预测结果
print("Predicted class:", predicted_class)