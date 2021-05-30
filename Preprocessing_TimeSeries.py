# -*- coding: utf-8 -*-
"""
Created on Fri May 28 10:05:32 2021

@author: maje8004
"""

import pandas as pd


compensate=pd.read_csv("compensate_info_global_increment.csv",sep='\t')
service=pd.read_csv("service_ticket_global.csv",sep='\t')

"""service_ticket_global preprocessing """
# service_ticket_global preprocessing (local_create_time)
# new date format: yyyy-mm-dd
service["local_create_time"]=service["local_create_time"].replace(regex=r' ',value="/")
split_date=service["local_create_time"].str.split("/",expand=True)
split_date["local_create_timev2"]=split_date.iloc[:,2].astype(str)+"-"+split_date.iloc[:,1].astype(str)+"-"+split_date.iloc[:,0].astype(str)+" "+split_date.iloc[:,3].astype(str)
service["local_create_time_v2"]=split_date["local_create_timev2"]
service=service.drop(['local_create_time'],axis=1)
service=service.rename(columns={"local_create_time_v2":"local_create_time"})
service.to_csv("service_ticket_global_py.csv",index=False)


"""4.0 Forecast for the next six weeks, after the last date of the data, the sum of the price compensations and the number of tickets that we will recieve"""
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


daily_price_tickets=pd.read_csv("daily_price_tickets.csv", index_col="date_service_ticket")#set date column as index
daily_price_tickets.index=daily_price_tickets.index.map(pd.to_datetime)# map to_datetime function to index
daily_price_tickets["Total_price"]=daily_price_tickets["Total_price"].replace(np.nan,0) #replace na with 0

price_series=daily_price_tickets[daily_price_tickets["Total_price"]!=0].Total_price
ticket_series=daily_price_tickets.Total_tickets

#lowly correlate with each other
corr_matrix=daily_price_tickets.corr(method="pearson")
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.show()

"""4.1 Price series"""
"""4.1.1 Decomposition"""

"""
#Trend: Increasing trend the first 15 days but the final 15 days of january price decreases and again 
at the beginning of february there is an increasing trend

#Seasonality: There is a seasonal pattern iN price series

"""
sns.lineplot(data=price_series)
decomposition_series=sm.tsa.seasonal_decompose(price_series,model="additive")
decomposition_series.plot()
plt.show()


"""4.1.2 Stationarity Test"""
### Stationarity test: Dickey-Fuller
#Ho: the series is not stationary, it presents heterocedasticity. In another words, your residue depends on itself (i.e.: yt depends on yt-1, yt-1 depends on yt-2 ..., and so on)
#Ha: the series is stationary (That is normally what we desire in regression analysis). Nothing more is needed to be done.
# p-value of 0.35>0.05(if we take 5% significance level or 95% confidence interval), null hypothesis cannot be rejected.

from statsmodels.tsa.stattools import adfuller
df_test=adfuller(price_series)
print("P-value: "+str(df_test[1])+" is greater than 0.05: reject Ho and conclude that series is non stationary (transformation for series is required)")

"""4.1.2.1 Transform orginal series"""
#First order Difference log transform
"""log(1+ri)=log(pi)-log(pj) -----> i: current row / j: previous row"""
price_series_log=np.log(price_series)-np.log((price_series.shift(1)))
price_series_log=price_series_log.dropna()
df_test=adfuller(price_series_log)
print("P-value: "+str(df_test[1])+" is lower than 0.05: reject Ho and conclude that series is stationary (no additional transform is required)")

sns.lineplot(data=price_series_log)

###seasonal component is observed
decomposition_series=sm.tsa.seasonal_decompose(price_series_log,model="additive")
decomposition_series.plot()
plt.show()


"""4.1.3 Split series into Train set and test set"""
#ratio_test=0.15
#ratio_train=0.85

y=price_series_log
Y_Train=y[:int(y.shape[0]*0.85)]
Y_Test=y[int(y.shape[0]*0.85):]

"""4.1.4 Fit Model"""
#We perform SARIMA wich is an extension of ARIMA that supports time series with seasonal component
"""
AR(p): regression model that uses the depndent relationship between an observation and lagged observation
I(d): differencing order to make time series stationary
MA(q): the model uses the dependency between observation and residual error from a moving average model (applied to lagged observations)

+

S(P,D,Q)m: are additional set of parameters that describe the seasonal components of the model

"""

#perform a auto arima first (to find the best parameters for arima model) 
#m=7 (daily data) https://alkaline-ml.com/pmdarima/tips_and_tricks.html#period
#trend="c" (constant)
#seasonal=True (Seasonal ARIMA)
# trace=True (printa status of the fits)
#Best model:  ARIMA(3,1,0)(2,0,0)[7]          
import pmdarima as pm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


#Auto ARIMA (Fit best parameters)
auto_model=pm.auto_arima(Y_Train, m=7, trend='c', seasonal=True,
                    test='adf',
                    stepwise=True, trace=True)

#Fit ARIMA model with parameters from Auto ARIMA
"""freq : str {'B','D','W','M','A', 'Q'}
    'B' - business day, ie., Mon. - Fri.
    'D' - daily
    'W' - weekly
    'M' - monthly
    'A' - annual
    'Q' - quarterly"""
    
#arima_model=ARIMA(Y_Train,order=(1,0,0),freq='D')
sarimamodel=SARIMAX(Y_Train,order=(3,1,0),seasonal_order=(2,0,0,7),freq='D')
results=sarimamodel.fit()
results.summary()
results.plot_diagnostics()


"""4.1.5 Prediction and Forecast"""
predictions_train=results.predict()  #estimated train data
predictions_test=results.predict(start=Y_Test.index[0],end=Y_Test.index[-1])  #estimated test data

from datetime import timedelta
forecast_price=results.predict(start=Y_Test.index[-1]+timedelta(days=1),end=Y_Test.index[-1]+timedelta(days=44)) #forecast of the following 6 weeks (42 days)
confidence_interval=results.conf_int()

Y_Train.plot() #Real data from train set
Y_Test.plot() #real data from test set
forecast_price.plot() #Forecast of next 6 weeks
predictions_train.plot()
confidence_interval.plot()
plt.show()


"""4.2 Ticekts series"""
"""4.2.1 Decomposition"""

"""
#Trend: decreasing trend the first 22 days but  after this point there is an increasing trend
#Seasonality: There is a seasonal pattern iN ticket series

"""
sns.lineplot(data=ticket_series)
decomposition_series=sm.tsa.seasonal_decompose(ticket_series,model="additive")
decomposition_series.plot()
plt.show()


"""4.2.2 Stationarity Test"""
### Stationarity test: Dickey-Fuller
#Ho: the series is not stationary, it presents heterocedasticity. In another words, your residue depends on itself (i.e.: yt depends on yt-1, yt-1 depends on yt-2 ..., and so on)
#Ha: the series is stationary (That is normally what we desire in regression analysis). Nothing more is needed to be done.
# p-value of 0.35>0.05(if we take 5% significance level or 95% confidence interval), null hypothesis cannot be rejected.

df_test=adfuller(ticket_series)
print("P-value: "+str(df_test[1])+" is greater than 0.05: reject Ho and conclude that series is non stationary (transformation for series is required)")

"""4.2.2.1 Transform orginal series"""
#First order Difference transform
ticket_series_log=np.log(ticket_series)-np.log((ticket_series.shift(1)))
ticket_series_log=ticket_series_log.dropna()
df_test=adfuller(ticket_series_log)
print("P-value: "+str(df_test[1])+" is lower than 0.05: reject Ho and conclude that series is stationary (no additional transform is required)")

sns.lineplot(data=price_series_log)

decomposition_series=sm.tsa.seasonal_decompose(ticket_series_log,model="additive")
decomposition_series.plot()
plt.show()


"""4.2.3 Split series intoTrain set and test set"""

y=ticket_series_log
Y_Train=y[:int(y.shape[0]*0.85)]
Y_Test=y[int(y.shape[0]*0.85):]

"""4.2.4 Fit Model"""
#We perform SARIMA model 

#perform a auto arima first (to find the best parameters for arima model) 
#m=7 (daily data) https://alkaline-ml.com/pmdarima/tips_and_tricks.html#period
#trend="c" (constant)
#seasonal=True (Seasonal ARIMA)
# trace=True (printa status of the fits)
#Best model:  ARIMA(0,0,1)(2,0,0)[7] intercept
import pmdarima as pm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


#Auto ARIMA (Fit best parameters)
auto_model=pm.auto_arima(Y_Train, m=7, trend='c', seasonal=True,
                    test='adf',
                    stepwise=True, trace=True)

#Fit ARIMA model with parameters from Auto ARIMA
"""freq : str {'B','D','W','M','A', 'Q'}
    'B' - business day, ie., Mon. - Fri.
    'D' - daily
    'W' - weekly
    'M' - monthly
    'A' - annual
    'Q' - quarterly"""
    
#arima_model=ARIMA(Y_Train,order=(1,0,0),freq='D')
sarimamodel=SARIMAX(Y_Train,order=(0,0,1),seasonal_order=(2,0,0,7),freq='D')
results=sarimamodel.fit()
results.summary()
results.plot_diagnostics()


"""4.2.5 Prediction and Forecast"""
predictions_train=results.predict() #estimated train set using sarima model
predictions_test=results.predict(start=Y_Test.index[0],end=Y_Test.index[-1])#estimated test set using sarima model

from datetime import timedelta
forecast_tickets=results.predict(start=Y_Test.index[-1]+timedelta(days=1),end=Y_Test.index[-1]+timedelta(days=43)) #forecast of the following 6 weeks (42 days)

Y_Train.plot() #Real data from train set
Y_Test.plot() #real data from test set
forecast_tickets.plot() #Forecast of next 6 weeks
plt.show()


"""4.3 Reverse Transform"""
"""revserse_i=reverse_j*exp(Delta_ij)= -----> i: current row / j: previous row"""
forecast=pd.concat([forecast_price,forecast_tickets],axis=1)
forecast=forecast.rename(columns={0:'delta_price',1:'delta_tickets'})
forecast["Total_tickets"]=np.nan
forecast["Total_price"]=np.nan
forecast=forecast.loc[:,['Total_tickets', 'Total_price', 'delta_price', 'delta_tickets']]

ticket_series_log=ticket_series_log.rename('delta_tickets')
price_series_log=price_series_log.rename('delta_price')
series_log_series_normal=pd.concat([ticket_series,price_series,price_series_log,ticket_series_log],axis=1)

last_data=series_log_series_normal.iloc[[-2]]
forecast.iloc[0,3]=series_log_series_normal.iloc[-1,3]
forecast=pd.concat([last_data,forecast])

#llenar a partir de segundo renglon
for i in range(1,len(forecast)):
    forecast.iloc[i,1]=round(((forecast.iloc[i-1,1])*(np.exp(forecast.iloc[i,2]))).astype(float))
    forecast.iloc[i,0]=((forecast.iloc[i-1,0])*(np.exp(forecast.iloc[i,3]))).astype(float)
    print(i)

daily_price_tickets.Total_tickets.plot()
forecast.Total_tickets.plot()
plt.show()

daily_price_tickets.Total_price.plot()
forecast.Total_price.plot()
plt.show()

forecast.reset_index(inplace=True)
forecast.to_csv("6_week_forecast.csv",index=False)


