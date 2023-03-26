import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 加载数据集
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# 数据预处理
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 构建模型
#在机器学习中，图像通常表示为形状为(height, width, channels)的三维张量，
# 其中height和width表示图像的高度和宽度，channels表示图像的通道数。
# 在这个示例中，(32, 32, 3) 表示输入到神经网络中的图像的形状是32x32像素，有RGB三个通道。

model = keras.Sequential(
    [
        keras.Input(shape=(32, 32, 3)),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10),
    ]
)

# 编译模型
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(lr=0.001),
    metrics=["accuracy"],
)

# 训练模型
model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.2)

# 评估模型
model.evaluate(x_test, y_test, batch_size=64, verbose=2)

model.save('./model/cifar10/cifar10')

# 这个模型是一个简单的卷积神经网络模型，用于对CIFAR-10数据集中的图像进行分类。
# 模型的架构包含了两个卷积层、两个池化层、一个全连接层和一个输出层。
# 在训练过程中，使用了RMSprop优化器和交叉熵损失函数。模型的评估使用了测试集数据。