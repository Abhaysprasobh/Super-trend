import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

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

# Handle NaN values
X = np.nan_to_num(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# PyTorch Dataset & DataLoader
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(pd.factorize(y)[0], dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = StockDataset(X_train, y_train)
test_dataset = StockDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# Initialize model
model = LSTMModel(input_dim=X_train.shape[2], hidden_dim=50, output_dim=3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Train the model
epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

# Evaluate model
model.eval()
y_pred_probs = torch.softmax(model(torch.tensor(X_test, dtype=torch.float32)), dim=1)
y_pred = torch.argmax(y_pred_probs, dim=1).numpy()
accuracy = np.mean(y_pred == pd.factorize(y_test)[0])
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Predict next day's stock volatility
def predict_next_day_volatility(model, last_30_days):
    model.eval()
    X_new = torch.tensor(last_30_days, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        prediction = torch.softmax(model(X_new), dim=1)
    prediction_label = ['Low', 'Medium', 'High'][torch.argmax(prediction).item()]
    print(f"Predicted Volatility for Next Day: {prediction_label}")
    return prediction_label
