import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import joblib

# Download historical data for multiple stocks
tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'TSLA', 'SPY']
start_date = '2015-01-01'
end_date = '2023-01-01'

data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')

# Feature engineering function
def compute_features(df, ticker):
    df = df.copy()
    df['Ticker'] = ticker
    df['Returns'] = df['Close'].pct_change()
    
    # Volatility
    df['Volatility_30'] = df['Returns'].rolling(30).std()
    
    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(14).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    sma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    df['Bollinger_%b'] = (df['Close'] - (sma20 - 2*std20)) / ((sma20 + 2*std20) - (sma20 - 2*std20))
    
    # Normalized features
    df['ATR_14_Pct'] = df['ATR_14'] / df['Close']
    df['MA_50_200_Ratio'] = df['Close'].rolling(50).mean() / df['Close'].rolling(200).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    # Momentum
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

# Define target variable using cross-sectional volatility tertiles
tertiles = full_df['Volatility_30'].quantile([0.33, 0.66]).values
full_df['Volatility_Category'] = pd.cut(full_df['Volatility_30'], 
                                      bins=[-np.inf, tertiles[0], tertiles[1], np.inf], 
                                      labels=['Low', 'Medium', 'High'])

# Feature selection
features = ['RSI_14', 'MACD', 'MACD_Signal', 'Bollinger_%b', 
            'ATR_14_Pct', 'MA_50_200_Ratio', 'Volume_Ratio',
            'Momentum_1M', 'Momentum_3M', 'Ticker']

# Preprocessing
X = pd.get_dummies(full_df[features], columns=['Ticker'])
y = full_df['Volatility_Category']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train Random Forest classifier
clf = RandomForestClassifier(n_estimators=150, 
                            max_depth=8,
                            class_weight='balanced',
                            random_state=42)
clf.fit(X_train, y_train)

# Evaluate model
y_pred = clf.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(clf, 'volatility_classifier.pkl')

# Adaptive Supertrend implementation
def adaptive_supertrend(df, model, lookback=30):
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Compute features for new data
    df = compute_features(df, 'Current')
    
    # Prepare features for prediction
    latest = df.iloc[[-1]].copy()
    latest_features = pd.get_dummies(latest[features], columns=['Ticker'])
    
    # Ensure all ticker columns are present
    for col in X.columns:
        if col not in latest_features.columns:
            latest_features[col] = 0
    
    # Predict volatility category
    prediction = model.predict(latest_features[X.columns])[0]
    
    # Set parameters based on volatility
    if prediction == 'Low':
        multiplier = 2.0
        period = 10
    elif prediction == 'Medium':
        multiplier = 3.0
        period = 14
    else:
        multiplier = 4.0
        period = 20
    
    # Calculate ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    
    # Calculate Supertrend
    hl2 = (df['High'].iloc[-1] + df['Low'].iloc[-1])/2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Generate signal
    close = df['Close'].iloc[-1]
    prev_close = df['Close'].iloc[-2] if len(df) > 1 else close
    
    if close > upper_band:
        signal = 1  # Buy
    elif close < lower_band:
        signal = -1  # Sell
    else:
        signal = 1 if prev_close > df['Close'].iloc[-1] else -1
    
    return {
        'volatility_category': prediction,
        'supertrend_upper': upper_band,
        'supertrend_lower': lower_band,
        'signal': signal,
        'multiplier': multiplier,
        'atr_period': period
    }

# Example usage
if __name__ == "__main__":
    # Load trained model
    model = joblib.load('volatility_classifier.pkl')
    
    # Get recent data for a stock
    test_data = yf.download('AAPL', period='1y', interval='1d')
    
    # Generate adaptive Supertrend signal
    analysis = adaptive_supertrend(test_data, model)
    
    print("\nAdaptive Supertrend Analysis:")
    print(f"Volatility Category: {analysis['volatility_category']}")
    print(f"Upper Band: {analysis['supertrend_upper']:.2f}")
    print(f"Lower Band: {analysis['supertrend_lower']:.2f}")
    print(f"Trading Signal: {'Buy' if analysis['signal'] == 1 else 'Sell'}")