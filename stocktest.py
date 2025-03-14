
import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

# Download historical stock data
tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'TSLA', 'SPY']
start_date = '2002-01-01'
end_date = '2023-01-01'
data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')

# Feature engineering function
def compute_features(df, ticker):
    df = df.copy()
    df['Ticker'] = ticker
    df['Returns'] = df['Close'].pct_change()
    df['Volatility_30'] = df['Returns'].rolling(30).std()
    
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(14).mean()
    
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    sma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    df['Bollinger_%b'] = (df['Close'] - (sma20 - 2*std20)) / ((sma20 + 2*std20) - (sma20 - 2*std20))
    
    df['ATR_14_Pct'] = df['ATR_14'] / df['Close']
    df['MA_50_200_Ratio'] = df['Close'].rolling(50).mean() / df['Close'].rolling(200).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    df['Momentum_1M'] = df['Close'].pct_change(21)
    df['Momentum_3M'] = df['Close'].pct_change(63)
    
    return df.dropna()

# Process data for all tickers
all_data = []
for ticker in tickers:
    try:
        df = data[ticker].copy()
        processed = compute_features(df, ticker)
        all_data.append(processed)
    except KeyError:
        continue

full_df = pd.concat(all_data)

# Define target variable using volatility tertiles
tertiles = full_df['Volatility_30'].quantile([0.33, 0.66]).values
full_df['Volatility_Category'] = pd.cut(full_df['Volatility_30'],
                                        bins=[-np.inf, tertiles[0], tertiles[1], np.inf],
                                        labels=['Low', 'Medium', 'High'])

# Feature selection
features = ['RSI_14', 'MACD', 'MACD_Signal', 'Bollinger_%b', 'ATR_14_Pct',
            'MA_50_200_Ratio', 'Volume_Ratio', 'Momentum_1M', 'Momentum_3M']

# Scaling features
scaler = StandardScaler()
full_df[features] = scaler.fit_transform(full_df[features])

# Prepare time-series data for LSTM
sequence_length = 30  # 30-day lookback window
X, y = [], []
for i in range(sequence_length, len(full_df)):
    X.append(full_df[features].iloc[i-sequence_length:i].values)
    y.append(full_df['Volatility_Category'].iloc[i])
X, y = np.array(X), np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Load the trained Keras model
model = load_model("stock.h5")  # Load the model saved in .h5 format

# Make predictions
y_pred_probs = model.predict(X_test)  # Keras model expects input in batch form
y_pred = np.argmax(y_pred_probs, axis=1)  # Get the class with highest probability

# Calculate accuracy
accuracy = np.mean(y_pred == pd.factorize(y_test)[0])
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Plot Predictions vs Actual
category_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
y_test_labels = [category_mapping[label] for label in pd.factorize(y_test)[0]]
y_pred_labels = [category_mapping[label] for label in y_pred]

plt.figure(figsize=(10, 5))
plt.plot(y_test_labels, label="Actual", marker='o', alpha=0.7)
plt.plot(y_pred_labels, label="Predicted", marker='x', linestyle='dashed', alpha=0.7)
plt.legend()
plt.title("LSTM Model Predictions vs Actual Volatility")
plt.xlabel("Sample Index")
plt.ylabel("Volatility Category")
plt.show()
