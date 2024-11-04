import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# Load the data
data = pd.read_csv('brent_oil_prices.csv', parse_dates=['Date'], dayfirst=True)
data.set_index('Date', inplace=True)
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Fill missing values through forward fill
data['Price'].fillna(method='ffill', inplace=True)

# Check for outliers
plt.figure(figsize=(10, 5))
sns.boxplot(x=data['Price'])
plt.title('Boxplot of Brent Oil Prices')
plt.show()

# Plot the time series data
plt.figure(figsize=(15, 7))
plt.plot(data['Price'], label='Brent Oil Price')
plt.xlabel('Date')
plt.ylabel('Price (USD per barrel)')
plt.title('Historical Brent Oil Prices')
plt.legend()
plt.show()

# ACF and PACF plots
plot_acf(data['Price'], lags=50)
plot_pacf(data['Price'], lags=50)
plt.show()

# Check stationarity
result = adfuller(data['Price'])
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')

# Differencing to make the series stationary
data['Price_diff'] = data['Price'].diff().dropna()

# Fit ARIMA model
model = auto_arima(data['Price_diff'].dropna(), seasonal=False, trace=True)
model.summary()

# Fit model with ARIMA parameters
arima_model = ARIMA(data['Price'], order=(5, 1, 0))
arima_result = arima_model.fit()
print(arima_result.summary())

# Predictions
pred = arima_result.predict(start=len(data), end=len(data) + 30, typ='levels')

# Plot predictions
plt.figure(figsize=(15, 7))
plt.plot(data['Price'], label='Historical Brent Oil Price')
plt.plot(pred, label='ARIMA Predictions', color='red')
plt.xlabel('Date')
plt.ylabel('Price (USD per barrel)')
plt.title('Brent Oil Price Predictions')
plt.legend()
plt.show()

# Calculate metrics
mse = mean_squared_error(data['Price'][-30:], pred[:30])
mae = mean_absolute_error(data['Price'][-30:], pred[:30])
print(f'MSE: {mse}')
print(f'MAE: {mae}')

# Example: Identify significant events and their impact on oil prices
significant_events = {
    '2020-03-09': 'Oil Price War between Saudi Arabia and Russia',
    '2020-04-20': 'Brent Crude Futures Turn Negative'
}

for date, event in significant_events.items():
    date = pd.to_datetime(date)
    if date in data.index:
        prev_date = date - pd.Timedelta(days=1)
        if prev_date in data.index:
            price_change = data.loc[date]['Price'] - data.loc[prev_date]['Price']
            print(f"Event: {event} on {date.strftime('%Y-%m-%d')}")
            print(f"Price Change: {price_change:.2f}\n")
        else:
            print(f"Previous day's data not available for event: {event} on {date.strftime('%Y-%m-%d')}\n")
    else:
        print(f"Data not available for event: {event} on {date.strftime('%Y-%m-%d')}\n")