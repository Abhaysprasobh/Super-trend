# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Set Seaborn style
sns.set()

# Fetch Microsoft stock data (2 years)
msft = yf.Ticker('MSFT')
df = msft.history(period="6mo")  # Get last 2 years of data

# Ensure data is sufficient
if df.empty:
    raise ValueError("No stock data fetched. Please check ticker symbol or internet connection.")

# 🔹 Feature Engineering: Moving Averages
df['MA5'] = df['Close'].rolling(window=5).mean()
df['MA10'] = df['Close'].rolling(window=10).mean()
df['MA20'] = df['Close'].rolling(window=20).mean()

# 🔹 Volatility Calculation
df['Daily_Return'] = df['Close'].pct_change()  # Daily percentage return
df['Volatility_10'] = df['Daily_Return'].rolling(window=10).std()  # 10-day volatility
df['Volatility_20'] = df['Daily_Return'].rolling(window=20).std()  # 20-day volatility
df['Annualized_Volatility'] = df['Volatility_20'] * np.sqrt(252)  # Annualized volatility

# 🔹 Min & Max Volatility Calculation
df['Min_Volatility'] = df['Volatility_20'].rolling(window=20).min()  # Minimum volatility in 20 days
df['Max_Volatility'] = df['Volatility_20'].rolling(window=20).max()  # Maximum volatility in 20 days

# 🔹 Target Variable: 1 if Next Day Close is Higher, Else 0
df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

# Drop NaN values created by rolling calculations
df.dropna(inplace=True)

# 🔹 Define Features and Target
features = ['MA5', 'MA10', 'MA20', 'Volatility_10', 'Volatility_20', 'Annualized_Volatility', 'Min_Volatility', 'Max_Volatility']
X = df[features]
y = df['Target']

# 🔹 Normalize Data using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🔹 Split Dataset (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 🔹 Hyperparameter Tuning for KNN (Finding Best k)
param_grid = {'n_neighbors': np.arange(1, 50, 2)}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best k value
best_k = grid_search.best_params_['n_neighbors']
print(f"Optimal k value: {best_k}")

# 🔹 Train Optimized KNN Model
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)

# 🔹 Evaluate Model
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"KNN Model Accuracy: {accuracy:.2%}")
print(classification_report(y_test, y_pred))

# 🔹 Predict Next Day's Movement
latest_data = X_scaled[-1].reshape(1, -1)
prediction = knn.predict(latest_data)
print(f"Predicted movement for next day: {'Up' if prediction[0] == 1 else 'Down'}")

# 🔹 Visualization: Min & Max Volatility Representation
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Volatility_20'], label="20-Day Volatility", color="blue", alpha=0.6)
plt.plot(df.index, df['Min_Volatility'], label="Min Volatility (20-Day)", color="green", linestyle="dashed")
plt.plot(df.index, df['Max_Volatility'], label="Max Volatility (20-Day)", color="red", linestyle="dashed")
plt.title("Microsoft (MSFT) Volatility Representation (Min & Max)")
plt.legend()
plt.show()
