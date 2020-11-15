import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

##Reading in the data
df = pd.read_csv('/Users/Luther/Desktop/data_analysis/Predicting_the_stock_market/sphist.csv')###
df['Date'] = pd.to_datetime(df['Date'])

df.sort_values('Date', ascending = True, inplace = True)
df.reset_index(drop = True, inplace = True)

##Generating Indicators
df['avg_5_days'] = df['Close'].rolling(5).mean().shift(1)
df['avg_30_days'] = df['Close'].rolling(30).mean().shift(1)
df['avg_365_days'] = df['Close'].rolling(365).mean().shift(1)

df['std_5_days'] = df['avg_5_days'].rolling(5).std().shift(1)
df['std_365_days'] = df['avg_365_days'].rolling(5).std().shift(1)

df['avg_5/avg_365'] = df['avg_5_days']/df['avg_365_days']
df['std_5/std_365'] = df['std_5_days']/df['std_365_days']


##Splitting up the data
df = df[df["Date"] > datetime(year=1951, month=1, day=2)]
df.dropna(axis = 0, inplace = True)
    # Create train and test dataframes
train = df[df["Date"] < datetime(year=2013, month=1, day=1)]
test = df[df["Date"] >= datetime(year=2013, month=1, day=1)]

##Making predictions

features = df.columns
features = features.drop(['Close', 'High', 'Low', 
'Open', 'Volume', 'Adj Close', 'Date'])
model = LinearRegression()
model.fit(train[features], train['Close'])
predictions = model.predict(test[features])
mae = mean_absolute_error(test['Close'], predictions)
mse = mean_squared_error(test['Close'], predictions)

print('mae:', mae)
print('mse:', mse)    