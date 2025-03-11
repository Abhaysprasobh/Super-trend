import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# List of stock tickers
tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'PYPL', 'ADBE', 'INTC',
    'NFLX', 'CMCSA', 'PEP', 'COST', 'TMUS', 'AVGO', 'TXN', 'QCOM', 'SBUX', 'AMD'
]

# Download historical OHLCV data for all tickers
data = yf.download(tickers, period='2y', group_by='ticker')

# Prepare features and calculate volatility
features = pd.DataFrame(index=tickers)
volatility_list = []

for ticker in tickers:
    df = data[ticker]
    
    # Calculate daily returns and volatility (annualized)
    returns = df['Close'].pct_change().dropna()
    volatility = returns.std() * np.sqrt(252)
    volatility_list.append(volatility)
    
    # Calculate average daily volume
    avg_volume = df['Volume'].mean()
    
    # Calculate average daily price range (High - Low)
    avg_range = (df['High'] - df['Low']).mean()
    
    # Calculate average daily return
    avg_daily_return = returns.mean()
    
    # Store features
    features.loc[ticker, 'avg_volume'] = avg_volume
    features.loc[ticker, 'avg_range'] = avg_range
    features.loc[ticker, 'avg_daily_return'] = avg_daily_return

# Create labels based on volatility tertiles
volatility_series = pd.Series(volatility_list, index=tickers)
labels = pd.qcut(volatility_series, q=3, labels=['low', 'medium', 'high'])

# Drop any rows with missing values
features = features.dropna()

# Align labels with features
labels = labels[features.index]

# Split data into training and testing sets
X = features.values
y = labels.cat.codes  # Convert categories to numerical codes

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')

# Example prediction
example_ticker = 'AAPL'
if example_ticker in features.index:
    example_features = features.loc[example_ticker].values.reshape(1, -1)
    example_features_scaled = scaler.transform(example_features)
    predicted_label = knn.predict(example_features_scaled)[0]
    print(f'{example_ticker} is predicted to have {labels.cat.categories[predicted_label]} volatility.')
else:
    print(f'{example_ticker} not found in the dataset.')