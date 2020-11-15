'''In this project, we'll be working with data from the S&P500 Index. 
The S&P500 is a stock market index.

We'll be using historical data on the price of the S&P500 Index 
to make predictions about future prices. Predicting whether an index will go 
up or down will help us forecast how the stock market as a whole will perform.

In this mission, we'll be working with a csv file containing index prices. 
Each row in the file contains a daily record of the price of the S&P500 Index 
from 1950 to 2015. The dataset is stored in sphist.csv'''

import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

#Reading in the data
df = pd.read_csv('sphist.csv')
df['Date'] = pd.to_datetime(df['Date'])

df.sort_values('Date', ascending = True, inplace = True)
df.reset_index(drop = True, inplace = True)

#Generating Indicators
df['avg_5_days'] = df['Close'].rolling(5).mean().shift(1)
df['avg_30_days'] = df['Close'].rolling(30).mean().shift(1)
df['avg_365_days'] = df['Close'].rolling(365).mean().shift(1)

df['std_5_days'] = df['avg_5_days'].rolling(5).std().shift(1)
df['std_365_days'] = df['avg_365_days'].rolling(5).std().shift(1)

df['avg_5/avg_365'] = df['avg_5_days']/df['avg_365_days']
df['std_5/std_365'] = df['std_5_days']/df['std_365_days']


#Splitting up the data
df = df[df["Date"] > datetime(year=1951, month=1, day=2)]
df.dropna(axis = 0, inplace = True)
#Create train and test dataframes
train = df[df["Date"] < datetime(year=2013, month=1, day=1)]
test = df[df["Date"] >= datetime(year=2013, month=1, day=1)]

#Making predictions
features = df.columns
'''Next, we'll drop the below because these all contain knowledge of the future 
that you don't want to feed the model.'''
features = features.drop(['Close', 'High', 'Low', 
'Open', 'Volume', 'Adj Close', 'Date'])
model = LinearRegression()
model.fit(train[features], train['Close'])
predictions = model.predict(test[features])
test['predicted_close'] = predictions

'''It's recommended to use Mean Absolute Error, also called MAE, as an error 
metric, because it will shows how "close" we were to the price in intuitive 
terms. Mean Squared Error, or MSE, is an alternative that is more commonly used, 
but makes it harder to intuitively tell how far off we are from the true price 
because it squares the error'''

mae = mean_absolute_error(test['Close'], test['predicted_close'])
mse = mean_squared_error(test['Close'], test['predicted_close'])

print('mae:', mae)
print('mse:', mse)
print(test.tail())