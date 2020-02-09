# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 20:32:54 2020

Incorporating code from Data Preprocessing of updated data.ipynb
into a single script file

Source Database:
- M. McCracken and S. Ng "FRED-MD: A Monthly Database for Macroeconomic Research", Working Paper, 2015.
- https://s3.amazonaws.com/files.fred.stlouisfed.org/fred-md/monthly/current.csv

@author: pjrowe2012
"""
import pandas as pd
import numpy as np

import time

# Anaconda has all these packages
from statsmodels.tsa.stattools import adfuller #to check unit root in time series  

from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

url = 'https://s3.amazonaws.com/files.fred.stlouisfed.org/fred-md/monthly/current.csv'
bigmacro=pd.read_csv(url)
bigmacro=bigmacro.rename(columns={'sasdate':'Date'})
bigmacro=bigmacro.iloc[1:,]
bigmacro=bigmacro.iloc[:-1,]

ts=time.localtime()
day= time.strftime('%Y-%m-%d', ts)
bigmacro.to_csv('current '+day+'.csv')

Recession_periods=pd.read_csv('Recession_Periods.csv')
regime = np.append(Recession_periods['Regime'].values,np.array(['Normal']*11))
bigmacro.insert(loc=1, column="Regime", value=regime)

"""
Following the steps below to clean data and make it ready for feature selection process.

1. Remove the variables with missing observations
2. Add lags of the variables as additional features
3. Test stationarity of time series
4. Standardize the dataset
"""
#------------- ------------- ------------- ------------- -------------
#1. Remove the variables with missing observations
missing_colnames=[]
print(bigmacro.shape)

for i in bigmacro.drop(['Date','Regime'],axis=1):
    observations=len(bigmacro)-bigmacro[i].count()
    if (observations>10):
        print(i+':'+str(observations))
        missing_colnames.append(i)

bigmacro=bigmacro.drop(labels=missing_colnames, axis=1)
#  there are a few rows with missing values but they are at the end of the dataset, so there are no missing months
#  in dataset; 59 years and 10 months, starting 1/1/1959, ending 10/2018, or 718 months
bigmacro=bigmacro.dropna(axis=0)

#------------- ------------- ------------- ------------- -------------
# 2. Add lags of the variables as additional features

for col in bigmacro.drop(['Date', 'Regime'], axis=1):
    for n in [3,6,9,12,18]:
        bigmacro['{} {}M lag'.format(col, n)] = bigmacro[col].shift(n).ffill().values 

# 1 month ahead prediction
bigmacro["Regime"] = bigmacro["Regime"].shift(-1)

bigmacro=bigmacro.dropna(axis=0)
bigmacro.tail(1)
# now only goes to 8/2019, due to shifting regime back one month
# 710 columns vs. 118 data columns before adding lags, 5x118 => 590+120=710

#------------- ------------- ------------- ------------- -------------
# 3. Test stationarity of time series
""" Augmented Dickey-Fuller Test can be used to test for stationarity in macroeconomic 
time series variables. We will use adfuller function from statsmodels module in Python. 
"""
#check stationarity
threshold=0.01 #significance level
for column in bigmacro.drop(['Date','Regime'], axis=1):
    result=adfuller(bigmacro[column])
    if result[1]>threshold:
        print(column)
        bigmacro[column]=bigmacro[column].diff()  # replaces values with diff between current and prior row value
bigmacro=bigmacro.dropna(axis=0)

threshold=0.01 #significance level
for column in bigmacro.drop(['Date','Regime'], axis=1):
    result=adfuller(bigmacro[column])
    if result[1]>threshold:
        print(column)
        bigmacro[column]=bigmacro[column].diff()
bigmacro=bigmacro.dropna(axis=0)

threshold=0.01 #significance level
for column in bigmacro.drop(['Date','Regime'], axis=1):
    result=adfuller(bigmacro[column])
    if result[1]>threshold:
        print(column)
bigmacro=bigmacro.dropna(axis=0)    

#------------- ------------- ------------- ------------- -------------
# 4. Standardize the dataset

features=bigmacro.drop(['Date','Regime'],axis=1)
col_names=features.columns

scaler=StandardScaler()
scaler.fit(features)
standardized_features=scaler.transform(features)
# features have been centered and scaled

print(standardized_features.shape)
df = pd.DataFrame(data=standardized_features, columns=col_names)
df.insert(loc=0, column="Date", value=bigmacro['Date'].values)
df.insert(loc=1, column='Regime', value=bigmacro['Regime'].values)
df.shape

df.to_csv('current_cleaned '+day+'.csv', index=False)
