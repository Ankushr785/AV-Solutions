import pandas as pd
import numpy as np
import os

os.chdir('/home/ankushraut/Downloads/av_student_hunt_2017')

data = pd.read_csv('train_pCWxroh.csv')
step = int(0.75*len(data))
test_ = pd.read_csv('test_bKeE5T8.csv')

ts1 = data['Count']
import matplotlib.pyplot as plt
plt.plot(ts1)

from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=7)
    rolstd = pd.rolling_std(timeseries, window=7)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
   
#moving_avg = pd.rolling_mean(ts_log, 7)
#expwighted_avg = pd.ewma(ts_log, halflife=7)
ts1_diff = np.log(ts1) - np.log(ts1).shift()   #differencing
ts1_diff = ts1_diff.dropna()

#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(ts1_diff, nlags=20)
lag_pacf = pacf(ts1_diff, nlags=20, method='ols')

#Plot ACF: 

plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts1_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts1_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts1_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts1_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()


#now apply ARIMA forecast

from statsmodels.tsa.arima_model import ARIMA

#Combined model

X = np.log(ts1).values
train, test = X[0:10000], X[10000:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(3,1,2))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))

predict = []
for i in range(len(predictions)):
    predict.append(predictions[i][0])
    
tes = []
for i in range(len(test)):
    tes.append(test[i])

pre_al = []
for i in range(len(np.exp(predict))):
    pre_al.append(np.exp(predict)[i])
    
tes_al = []
for i in range(len(np.exp(tes))):
    tes_al.append(np.exp(tes)[i])


sse = 0
for i in range(len(tes_al)):
    sse+=(tes_al[i] - pre_al[i])**2
residuals = []
for i in range(len(tes_al)):
    residuals.append(tes_al[i] - pre_al[i])
    
rmse = (sse/len(tes_al))**0.5
print(rmse, np.mean(residuals))

# plot
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()
