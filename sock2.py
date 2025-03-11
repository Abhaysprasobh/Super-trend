import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Download data with explicit column handling
df = yf.download('MSFT', period='2y', auto_adjust=False)

# Validate data structure
if df.empty or 'Close' not in df.columns:
    raise ValueError("Data download failed or missing required columns")

# Manual Technical Indicator Calculations -------------------------------------------------

# 1. Calculate ATR (14-period)
def calculate_atr(df, window=14):
    high_low = df['High'] - df['Low']
    high_pclose = abs(df['High'] - df['Close'].shift(1))
    low_pclose = abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([high_low, high_pclose, low_pclose], axis=1).max(axis=1)
    atr = tr.rolling(window).mean().fillna(method='bfill')
    return atr

# 2. Calculate RSI (14-period)
def calculate_rsi(df, window=14):
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(method='bfill')

# 3. Calculate Moving Averages
def calculate_moving_averages(df):
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    return df

# 4. Calculate MACD
def calculate_macd(df):
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    return df

# Apply all calculations
df['ATR'] = calculate_atr(df)
df['RSI'] = calculate_rsi(df)
df = calculate_moving_averages(df)
df = calculate_macd(df)

# Create target variable
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df.dropna(inplace=True)

# Feature Engineering --------------------------------------------------------------------
features = ['ATR', 'RSI', 'SMA_20', 'EMA_50', 'MACD']
X = df[features]
y = df['Target']

# Time-series split (no shuffling)
train_size = int(0.8 * len(df))
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training -------------------------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42,
    min_samples_split=10,
    max_depth=10
)
model.fit(X_train_scaled, y_train)

# Model Evaluation -----------------------------------------------------------------------
y_pred = model.predict(X_test_scaled)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print(classification_report(y_test, y_pred))

# Feature Importance
importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(importance)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=["Down", "Up"], 
            yticklabels=["Down", "Up"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Next Day Prediction --------------------------------------------------------------------
latest_data = scaler.transform(X.tail(1))
prediction = model.predict(latest_data)
print(f"\nNext Day Prediction: {'Up' if prediction[0] == 1 else 'Down'}")