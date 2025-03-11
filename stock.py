# Import necessary libraries
import pandas as pd
import numpy as np
import yfinance as yf
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier

# Fetch Microsoft stock data (5 years)
print("Fetching Microsoft stock data...")
msft = yf.Ticker('MSFT')
df = msft.history(period="5y")

# Check if data is fetched successfully
if df.empty:
    raise ValueError("No data fetched. Check the ticker symbol or internet connection.")

# Feature Engineering
print("Creating features...")
# Moving Averages
df['MA5'] = df['Close'].rolling(window=5).mean()  # 5-day moving average
df['MA10'] = df['Close'].rolling(window=10).mean()  # 10-day moving average
df['MA20'] = df['Close'].rolling(window=20).mean()  # 20-day moving average

# Volatility (Average True Range - ATR)
df['High-Low'] = df['High'] - df['Low']
df['High-Close_prev'] = abs(df['High'] - df['Close'].shift(1))
df['Low-Close_prev'] = abs(df['Low'] - df['Close'].shift(1))
df['True_Range'] = df[['High-Low', 'High-Close_prev', 'Low-Close_prev']].max(axis=1)
df['ATR'] = df['True_Range'].rolling(window=14).mean()  # 14-day ATR

# Momentum Indicators
df['MACD'] = df['MA10'] - df['MA20']  # Moving Average Convergence Divergence
df['RSI'] = 100 - (100 / (1 + (df['Close'].pct_change().rolling(14).mean() / df['Close'].pct_change().rolling(14).std())))  # Relative Strength Index

# Bollinger Bands
df['Bollinger_Upper'] = df['MA20'] + (2 * df['Close'].rolling(window=20).std())
df['Bollinger_Lower'] = df['MA20'] - (2 * df['Close'].rolling(window=20).std())

# Volume Weighted Average Price (VWAP)
df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()

# Target Variable: 1 if next day's close is higher, else 0
df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

# Drop rows with missing values (created by rolling calculations)
df.dropna(inplace=True)

# Define features and target
features = ['MA5', 'MA10', 'MA20', 'ATR', 'MACD', 'RSI', 'Bollinger_Upper', 'Bollinger_Lower', 'VWAP']
X = df[features]  # Features
y = df['Target']  # Target

# Balance the dataset using SMOTE (to handle imbalanced classes)
print("Balancing dataset with SMOTE...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Normalize the data using RobustScaler (handles outliers better)
print("Scaling features...")
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_resampled)

# Reduce dimensionality using PCA (retain 95% of variance)
print("Reducing dimensionality with PCA...")
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# Split the data into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X_pca, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Hyperparameter tuning for KNN
print("Tuning KNN hyperparameters...")
param_grid = {
    'n_neighbors': np.arange(1, 50, 2),  # Test odd numbers from 1 to 49
    'weights': ['uniform', 'distance'],  # Weighting strategy
    'metric': ['euclidean', 'manhattan', 'minkowski']  # Distance metric
}

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),  # 10-fold cross-validation
    scoring='accuracy',
    n_jobs=-1  # Use all CPU cores
)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print(f"Best KNN Parameters: {best_params}")

# Train the KNN model with the best hyperparameters
print("Training KNN model...")
knn = KNeighborsClassifier(
    n_neighbors=best_params['n_neighbors'],
    weights=best_params['weights'],
    metric=best_params['metric']
)
knn.fit(X_train, y_train)

# Evaluate the model
print("Evaluating model...")
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"KNN Model Accuracy: {accuracy:.2%}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Ensemble Model (KNN + XGBoost)
print("Training ensemble model...")
xgb = XGBClassifier(random_state=42, n_estimators=100, learning_rate=0.1)
ensemble = VotingClassifier(
    estimators=[('knn', knn), ('xgb', xgb)],
    voting='soft'  # Use soft voting for probability-based predictions
)
ensemble.fit(X_train, y_train)

# Evaluate the ensemble model
y_pred_ensemble = ensemble.predict(X_test)
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
print(f"Ensemble Model Accuracy: {accuracy_ensemble:.2%}")
print("Ensemble Classification Report:")
print(classification_report(y_test, y_pred_ensemble))

# Predict the next day's movement
latest_data = X_pca[-1].reshape(1, -1)  # Use the latest data point
prediction = ensemble.predict(latest_data)
print(f"Predicted movement for next day: {'Up' if prediction[0] == 1 else 'Down'}")