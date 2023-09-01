import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow import keras

# Fetch stock prices data (e.g., Apple)
import yfinance as yf
data = yf.download('AAPL', start="2010-01-01", end="2020-12-31")
prices = data['Close'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler()
prices_normalized = scaler.fit_transform(prices)

# Prepare the data for LSTM
X = [prices_normalized[i:i+60] for i in range(len(prices_normalized)-60)]
y = [prices_normalized[i+60] for i in range(len(prices_normalized)-60)]

X = np.array(X)
y = np.array(y)

# Split the data into train and test
train_size = int(0.8 * len(X))
X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

# LSTM Model
model = keras.models.Sequential([
    keras.layers.LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    keras.layers.LSTM(50),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Transform predictions back to original scale
y_pred_train_original = scaler.inverse_transform(y_pred_train)
y_pred_test_original = scaler.inverse_transform(y_pred_test)
y_train_original = scaler.inverse_transform(y_train)
y_test_original = scaler.inverse_transform(y_test)

# Visualize the results
plt.figure(figsize=(15, 6))
plt.plot(np.arange(len(y_train_original)), y_train_original, color='blue', label='Actual train')
plt.plot(np.arange(len(y_pred_train_original)), y_pred_train_original, color='red', label='Predicted train')
plt.plot(np.arange(len(y_train_original), len(y_train_original) + len(y_test_original)), y_test_original, color='green', label='Actual test')
plt.plot(np.arange(len(y_pred_train_original), len(y_pred_train_original) + len(y_pred_test_original)), y_pred_test_original, color='orange', label='Predicted test')
plt.legend()
plt.show()
