import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def fetch_stock_data(ticker, period="2y"):
    """Fetch stock data from Yahoo Finance"""
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    if df.empty:
        raise ValueError("No stock data fetched. Check ticker symbol or internet connection.")
    return df

def compute_technical_indicators(df):
    """Compute moving averages, RSI, MACD, and Bollinger Bands"""
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['RSI14'] = compute_rsi(df)
    
    # MACD Calculation
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    
    # Bollinger Bands
    df['BB_Upper'] = df['MA20'] + (df['Close'].rolling(20).std() * 2)
    df['BB_Lower'] = df['MA20'] - (df['Close'].rolling(20).std() * 2)
    
    df['Volatility'] = df['Close'].pct_change().rolling(window=10).std()
    df.dropna(inplace=True)
    return df

def compute_rsi(data, window=14):
    """Calculate the Relative Strength Index (RSI)"""
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def prepare_lstm_data(df, sequence_length=60):
    """Prepare time series data for LSTM"""
    features = ['Close', 'MA10', 'MA20', 'RSI14', 'MACD', 'BB_Upper', 'BB_Lower', 'Volatility']
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])
    
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i+sequence_length])
        y.append(scaled_data[i+sequence_length, 0])  # Predicting 'Close' price
    
    return np.array(X), np.array(y), scaler

def build_lstm_model(input_shape):
    """Build an optimized Bidirectional LSTM model"""
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True, activation='relu', kernel_regularizer='l2'), input_shape=input_shape),
        Dropout(0.3),
        Bidirectional(LSTM(64, return_sequences=False, activation='relu', kernel_regularizer='l2')),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

# Main Execution
if __name__ == "__main__":
    df = fetch_stock_data('MSFT')
    df = compute_technical_indicators(df)
    
    sequence_length = 60  # Use last 60 days for prediction
    X, y, scaler = prepare_lstm_data(df, sequence_length)
    
    split = int(len(X) * 0.8)  # 80% Training, 20% Testing
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
    
    model = build_lstm_model((sequence_length, X.shape[2]))
    
    # Early Stopping & Learning Rate Reduction
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
    
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), 
              callbacks=[early_stop, lr_reduce])
    
    # Predict next day's price
    last_sequence = X[-1].reshape(1, sequence_length, X.shape[2])
    predicted_price = model.predict(last_sequence)
    predicted_price = scaler.inverse_transform([[predicted_price[0][0], 0, 0, 0, 0, 0, 0, 0]])[0][0]
    
    print(f"Predicted next day's closing price: {predicted_price:.2f}")
