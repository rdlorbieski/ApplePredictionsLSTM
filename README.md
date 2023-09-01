
# Stock Price Forecasting using LSTM


This project demonstrates a simple application of LSTM (Long Short-Term Memory) neural networks to forecast stock prices using historical closing price data.



## Description

The code fetches historical daily closing prices for Apple (AAPL) from 2010 to 2020 using the **"yfinance"** library. The LSTM network is trained using the past 60 days of prices to predict the next day's closing price. The data is normalized to a range between 0 and 1 before training, and predictions are transformed back to the original price scale for visualization and evaluation.


## Dependencies

- numpy
- pandas
- matplotlib
- scikit-learn
- tensorflow
- yfinance
## Usage

Simply run the code to fetch the stock data, train the LSTM model, and visualize the actual vs. predicted prices on both the training and test datasets.


## Installing

Remember to install the required libraries using pip:


```bash
  pip install numpy pandas matplotlib scikit-learn tensorflow yfinance
```
    