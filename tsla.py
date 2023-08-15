"""
			Stock Price Prediction:
	
	Author: Howard Anderson.
	
	Date: 02/07/2023.
	
	Filename: tsla.py
	
	Description: Prediction of Tesla's stocks using LSTM.

"""

# Importing necessary Libraries.
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Imports from TensorFlow.
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# Taking Close as feature to Predict.
data = pd.read_csv("TSLA.csv")
train = data.iloc[:, 4:5].values

# Spliting Training and Testing data.
train = train[:-121]
test = train[-120:]

# Scaler: MinMaxScaler - Standardizing the data.
scaler = MinMaxScaler(feature_range = (0,1))
train = scaler.fit_transform(train)

# Preprocessing: Making Time-Series vaults..
x_train = []
y_train = []
x_test = []
y_test = []


for index in range(0, len(train) - 60):
	x_train.append(train[index:index+60])
	y_train.append(train[index+60:index+61])


for index in range(0, len(test) - 60):
	x_test.append(test[index:index+60])
	y_test.append(test[index+59:index+60])
#	print(f"Index = {index}, Index + 59 = {index + 60} ")


# Preprocessing: Reshaping NP.ARRAY for LSTM.
x_train = np.array(x_train, dtype = "float32")
y_train = np.array(y_train, dtype = "float32")
x_test = np.array(x_test, dtype = "float32")
y_test = np.array(y_test, dtype = "float32")

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# Model Definition.
predictor = Sequential()
predictor.add(LSTM(units = 100, return_sequences = True, input_shape = (x_train.shape[1],1)))
predictor.add(Dropout(0.2))
predictor.add(LSTM(units = 100, return_sequences = True))
predictor.add(Dropout(0.2))
predictor.add(LSTM(units = 100, return_sequences = True))
predictor.add(Dropout(0.2))
predictor.add(LSTM(units = 100, return_sequences = False))
predictor.add(Dropout(0.2))
predictor.add(Dense(units = 1))

predictor.summary()

"""
# Setting Model Hyper-Parameters.
predictor.compile(
				optimizer = "adam",
				loss = "mean_squared_error",
				metrics = ["accuracy"],
				)

# Training the Model.
predictor.fit(x_train, y_train, epochs = 400)

loss, acc = predictor.evaluate(x_test, y_test, verbose = 2)
predictions = predictor.predict(x_test)
print(predictions[5])

#print(f"\nLoss: {loss}, Accuracy: {acc}\n")

"""
