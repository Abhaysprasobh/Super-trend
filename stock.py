# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import mstats

# Set Seaborn style
sns.set()

# Fetch Microsoft stock data (5 years)
print("Fetching Microsoft stock data...")
msft = yf.Ticker('AAPL')
df = msft.history(period="2y", auto_adjust=True)

# Ensure data is sufficient
if df.empty:
    raise ValueError("No stock data fetched. Check ticker symbol or internet connection.")

# 🔹 Moving Averages (Trend Indicators)
df['MA5'] = df['Close'].rolling(window=5).mean()
df['MA10'] = df['Close'].rolling(window=10).mean()
df['MA20'] = df['Close'].rolling(window=20).mean()

# 🔹 ATR Calculation (Volatility & Trend Direction)
df['High-Low'] = df['High'] - df['Low']
df['High-Close_prev'] = abs(df['High'] - df['Close'].shift(1))
df['Low-Close_prev'] = abs(df['Low'] - df['Close'].shift(1))
df['True_Range'] = df[['High-Low', 'High-Close_prev', 'Low-Close_prev']].max(axis=1)
df['ATR'] = df['True_Range'].ewm(span=14, adjust=False).mean()

# 🔹 ATR Trend Signal (Identifies Trending vs Ranging Market)
df['ATR_Change'] = df['ATR'].pct_change()
df['ATR_Trend'] = np.where(df['ATR_Change'] > 0, 1, 0)  # 1 = Trending, 0 = Ranging

# 🔹 Volatility Measures
df['Daily_Return'] = df['Close'].pct_change()
df['Volatility_10'] = df['Daily_Return'].rolling(window=10).std()
df['Volatility_20'] = df['Daily_Return'].rolling(window=20).std()
df['Annualized_Volatility'] = df['Volatility_20'] * np.sqrt(252)

# 🔹 Momentum Indicators
df['MACD'] = df['MA10'] - df['MA20']
df['RSI'] = 100 - (100 / (1 + df['Close'].pct_change().rolling(14).mean() / df['Close'].pct_change().rolling(14).std()))

# 🔹 Bollinger Bands
df['Bollinger_Upper'] = df['MA20'] + (df['ATR'] * 2)
df['Bollinger_Lower'] = df['MA20'] - (df['ATR'] * 2)

# 🔹 VWAP (Volume Weighted Average Price)
df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()

# 🔹 Target Variable: 1 if Next Day Close is Higher, Else 0
df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

# Drop NaN values created by rolling calculations
df.dropna(inplace=True)

# 🔹 Handle Outliers Using Winsorization
for col in ['ATR', 'Volatility_10', 'Volatility_20', 'MACD', 'RSI']:
    df[col] = mstats.winsorize(df[col], limits=[0.01, 0.01])  # Removes extreme values

# 🔹 Define Features and Target
features = ['ATR', 'ATR_Trend', 'Volatility_10', 'Volatility_20', 'Annualized_Volatility', 'MACD', 'RSI', 'VWAP', 'Bollinger_Upper', 'Bollinger_Lower']
X = df[features]
y = df['Target']

# 🔹 Normalize Data using RobustScaler (Better for Outliers)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# 🔹 Reduce Dimensionality using PCA (Retains 95% Variance)
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# 🔹 Stratified Train-Test Split (Balances Class Distribution)
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)

# 🔹 Hyperparameter Tuning for KNN
param_grid = {
    'n_neighbors': np.arange(3, 15, 2),  # Best k-range for stock data
    'weights': ['uniform', 'distance'],
    'metric': ['manhattan', 'minkowski']  # Avoiding 'euclidean' for better similarity measures
}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=StratifiedKFold(n_splits=10), scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get best KNN parameters
best_params = grid_search.best_params_
print(f"Best KNN Parameters: {best_params}")

# 🔹 Train Optimized KNN Model
knn = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'], metric=best_params['metric'], weights=best_params['weights'])
knn.fit(X_train, y_train)

# 🔹 Evaluate Model
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"KNN Model Accuracy: {accuracy:.2%}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 🔹 Predict Next Day's Movement
latest_data = X_pca[-1].reshape(1, -1)
prediction = knn.predict(latest_data)
print(f"Predicted movement for next day: {'Up' if prediction[0] == 1 else 'Down'}")





