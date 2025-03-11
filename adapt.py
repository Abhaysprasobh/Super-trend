import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, ConfusionMatrixDisplay)
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

# Download historical stock data
ticker = 'AAPL'  # Example ticker
df = yf.download(ticker, start='2023-06-01', end='2023-12-31', progress=False)

# Calculate logarithmic returns
df['log_ret'] = np.log(df['Close']).diff()

# Calculate forward-looking volatility (target variable)
volatility_window = 10  # Volatility calculation window in days
df['future_vol'] = df['log_ret'].rolling(volatility_window).std().shift(-volatility_window)

# Debug: print tail to verify future_vol values
print("Before cleaning:")
print(df[['log_ret', 'future_vol']].tail(20))

# Ensure the future_vol column exists and drop the last volatility_window rows,
# since these rows have NaN due to the shift.
if 'future_vol' not in df.columns:
    raise ValueError("Error: 'future_vol' column is missing.")
    
# Remove the last N rows, where N equals volatility_window
df = df.iloc[:-volatility_window]

# Double-check that future_vol no longer has NaN values
if df['future_vol'].isnull().any():
    raise ValueError("Error: NaN values are still present in 'future_vol' after cleaning!")
    
print("Data cleaned successfully!")
print("After cleaning:")
print(df[['log_ret', 'future_vol']].tail(15))

# Create volatility classes using quantiles
df['label'] = pd.qcut(df['future_vol'], q=[0, 0.25, 0.75, 1], 
                      labels=['low', 'medium', 'high'])

# Feature Engineering
def calculate_features(data):
    # Historical volatility (20-day rolling)
    data['hist_vol_20'] = data['log_ret'].rolling(20).std()
    
    # RSI (14-day period)
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    data['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD features
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['macd'] = ema12 - ema26
    data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands features
    sma20 = data['Close'].rolling(20).mean()
    std20 = data['Close'].rolling(20).std()
    data['bollinger_pct'] = (data['Close'] - (sma20 - 2 * std20)) / ((sma20 + 2 * std20) - (sma20 - 2 * std20))
    
    # Momentum features
    data['momentum_10'] = data['Close'].pct_change(10)
    data['volume_ma5'] = data['Volume'].rolling(5).mean()
    
    return data

df = calculate_features(df)
# Remove any remaining rows with NaN values from feature calculations
df.dropna(inplace=True)

# Prepare features and target
features = ['hist_vol_20', 'rsi', 'macd', 'macd_signal', 
            'bollinger_pct', 'momentum_10', 'volume_ma5']
X = df[features]
y = LabelEncoder().fit_transform(df['label'])

# Time-based train-test split
split_idx = int(0.8 * len(X))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning with time-series cross-validation
param_grid = {
    'n_neighbors': [5, 7, 9, 11, 13, 15],
    'weights': ['uniform', 'distance'],
    'p': [1, 2],
    'leaf_size': [20, 30, 40]
}

knn = KNeighborsClassifier()
tscv = TimeSeriesSplit(n_splits=5)
grid_search = GridSearchCV(knn, param_grid, cv=tscv, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Best model evaluation
best_knn = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Validation Accuracy: {grid_search.best_score_:.3f}")

# Test set evaluation
y_pred = best_knn.predict(X_test_scaled)
test_acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {test_acc:.3f}")
print(classification_report(y_test, y_pred, target_names=['low', 'medium', 'high']))

# Confusion matrix visualization
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                              display_labels=['low', 'medium', 'high'])
disp.plot(cmap='Blues')
plt.title('Volatility Classification Confusion Matrix')
plt.show()

# Feature importance analysis
result = permutation_importance(best_knn, X_test_scaled, y_test, 
                                n_repeats=10, random_state=42)
sorted_idx = result.importances_mean.argsort()[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importance via Permutation")
plt.barh(range(X.shape[1]), result.importances_mean[sorted_idx])
plt.yticks(range(X.shape[1]), [features[i] for i in sorted_idx])
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()
