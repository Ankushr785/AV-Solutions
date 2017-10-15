import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt


import os
os.chdir('/home/ankushraut/Downloads/av_student_hunt_2017')

data = pd.read_csv('train_pCWxroh.csv')
step = int(0.75*len(data))
test = pd.read_csv('test_bKeE5T8.csv')

#frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag):
	df = pd.DataFrame(data)
	columns = [df.shift(lag)]
	columns.append(df)
	df = pd.concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return pd.Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# scale train and test data to [-1, 1]
def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = np.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]

# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()
	return model

# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]


series_d = data['Count']
#transform data to be stationary
raw_values_d = series_d.values
diff_values_d = difference(raw_values_d, 1)

lag = 1
        
# transform data to be supervised learning
supervised_d = timeseries_to_supervised(diff_values_d, lag)
supervised_values_d = supervised_d.values

# split data into train and test-sets
train_d, test_d = supervised_values_d[0:step], supervised_values_d[step:]

# transform the scale of the data
scaler_d, train_scaled_d, test_scaled_d = scale(train_d, test_d)

# fit the model
lstm_model_d = fit_lstm(train_scaled_d, 1, 500, 4)
# forecast the entire training dataset to build up state for forecasting
train_reshaped_d = train_scaled_d[:, 0].reshape(len(train_scaled_d), 1, 1)
train_fit_d = lstm_model_d.predict(train_reshaped_d, batch_size=1)

train_reshaped_d1 = []
for i in range(len(train_fit_d)):
    train_reshaped_d1.append(train_reshaped_d[i][0])

# walk-forward validation on the test data
predictions_d = list()
##
for i in range(len(test_scaled_d)):
    # make one-step forecast
    X_d, y_d = test_scaled_d[i, 0:-1], test_scaled_d[i, -1]
    yhat_d = forecast_lstm(lstm_model_d, 1, X_d)
    #invert scaling
    yhat_d = invert_scale(scaler_d, X_d, yhat_d)
    #invert differencing
    yhat_d = inverse_difference(raw_values_d, yhat_d, len(test_scaled_d)+1-i)
    #store forecast
    predictions_d.append(yhat_d)
    expected_d = raw_values_d[len(train_d) + i + 1]
    print('Day=%d, Predicted_d=%f, Expected_d=%f' % (i+1, yhat_d, expected_d))
    
# report performance
rmse = sqrt(mean_squared_error(raw_values_d[step+1:], predictions_d))
print('Test RMSE: %.3f' % rmse)

# line plot of observed vs predicted

plt.figure(figsize = (12,8))
plt.plot(train_reshaped_d1, color = 'blue', label = 'actual_values_scaled')
plt.plot(train_fit_d, color = 'red', label = 'fitted_values_scaled')
plt.ylabel('Demand')
plt.legend()
plt.title('Training fit on Demand')
plt.show()

#prediction graph
plt.figure(figsize=(12,8))
plt.plot(predictions_d, color = 'red', label = 'predicted_values')
plt.plot(raw_values_d[step+1:], color = 'blue', label = 'actual_values')
plt.legend()
plt.ylabel('Demand')
plt.title('Predictions of Demand')
plt.show()

forecast_d = []
for i in range(len(test)):
    forecast = forecast_lstm(lstm_model_d, 1, np.array([test_scaled_d[-1, -1]]))
    forecast_is = invert_scale(scaler_d, np.array([test_scaled_d[-1, -1]]), forecast)
    forecast_id = inverse_difference(raw_values_d, forecast_is, 1)
    test_scaled_d[:,-1]+=np.array([forecast])
    forecast_d.append(forecast_id)

submission = pd.DataFrame({'ID':test.ID, 'Count':np.round(forecast_d)})
submission.to_csv('submission.csv', index = False)