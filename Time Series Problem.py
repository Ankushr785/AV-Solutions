import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense
import os
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from sklearn.metrics import mean_squared_error

os.chdir('/home/Akai/Downloads/time_series_av')

data = pd.read_csv('train_av_time.csv')
test = pd.read_csv('test_av_time.csv')

#parameter setting
step = 1000
epochs = 100
neurons = 4
pred_range = 5112

#setting up LSTM environment  (doesn't need calibration)

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


#Fitting time series model


series = data['Count']
r = len(series) - step

#transform data to be stationary
raw_values = series.values
diff_values = difference(raw_values, 1)

#selecting the best lag
lag1 = series.shift()
lag2 = lag1.shift()
lag3 = lag2.shift()
lag4 = lag3.shift()
lag5 = lag4.shift()
lag6 = lag5.shift()
lag7 = lag6.shift()

lag_df = pd.DataFrame({'series':series, 'lag1':lag1, 'lag2':lag2, 'lag3':lag3, 'lag4':lag4, 'lag5':lag5, 'lag6':lag6, 'lag7':lag7})
lag_df = lag_df.drop(lag_df.index[[0, 1, 2, 3, 4, 5, 6]])

comparison = []
for i in range(7):
    comparison.append(np.corrcoef(lag_df.iloc[:, 0], lag_df.iloc[:, i+1])[0,1])

for i in range(7):
    if comparison[i] == np.max(comparison):
        lag = i+1
    else:
        count = 0

# transform data to be supervised learning
supervised = timeseries_to_supervised(diff_values, lag)
supervised_values = supervised.values

# split data into train and test-sets
train, test = supervised_values[0:-step], supervised_values[-step:]

# transform the scale of the data
scaler, train_scaled, test_scaled= scale(train, test)

# fit the model
lstm_model = fit_lstm(train_scaled, 1, epochs, neurons)
# forecast the entire training dataset to build up state for forecasting
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
train_fit = lstm_model.predict(train_reshaped, batch_size=1)

train_reshaped1 = []
for i in range(len(train_fit)):
    train_reshaped1.append(train_reshaped[i][0])

# walk-forward validation on the test data
predictions = list()
for i in range(len(test_scaled)):
    # make one-step forecast
    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    yhat = forecast_lstm(lstm_model, 1, X)
    #invert scaling
    yhat = invert_scale(scaler, X, yhat)
    #invert differencing
    yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
    #store forecast
    predictions.append(yhat)
    expected = raw_values[len(train) + i + 1]
    print('Day=%d, Predicted_d=%f, Expected_d=%f' % (i+1, yhat, expected))

# report performance
rmse = sqrt(mean_squared_error(raw_values[-step:], predictions))
print('Test RMSE: %.3f' % rmse)

# line plot of observed vs predicted
plt.figure(figsize = (12,8))
plt.plot(train_reshaped1[-30:], color = 'blue', label = 'actual_values_scaled')
plt.plot(train_fit[-30:], color = 'red', label = 'fitted_values_scaled')
plt.ylabel('Demand')
plt.legend()
plt.title('Training fit on Demand')
plt.show()

#prediction graph
plt.figure(figsize=(12,8))
plt.plot(predictions, color = 'red', label = 'predicted_values')
plt.plot(raw_values[r:], color = 'blue', label = 'actual_values')
plt.legend()
plt.ylabel('Demand')
plt.title('Predictions of Demand')
plt.show()

plt.figure(figsize=(12,8))
plt.plot(series)
plt.title('Sales Time Series')
plt.ylabel('Sales')
plt.xlabel('Time')
plt.show()

forecasted_values = []
for i in range(pred_range):
    forecast = forecast_lstm(lstm_model, 1, np.array([test_scaled[-1, -1]]))
    forecast_is = invert_scale(scaler, np.array([test_scaled[-1, -1]]), forecast)
    forecast_id = inverse_difference(raw_values, forecast_is, 1)
    test_scaled[:,-1] += np.array([forecast])
    forecasted_values.append(forecast_id)
    
predicted_values = []    
for i in range(len(predictions)):
    predicted_values.append(predictions[i])
    
test.Count = pd.DataFrame({'Count':np.round(predicted_values)})
test.to_csv('submission.csv', index=False)