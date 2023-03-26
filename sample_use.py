import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
import keras



model = keras.models.load_model('./model/sample/sample')

x_test = [10, 6.5, 2, -3, 1]
prediction = model.predict(x_test)
print(prediction)