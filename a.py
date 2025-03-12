import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Fetch stock data
def fetch_stock_data(ticker, period="2y"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    if df.empty:
        raise ValueError("No stock data fetched. Check ticker symbol or internet connection.")
    return df

# Compute technical indicators
def compute_technical_indicators(df):
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['RSI14'] = compute_rsi(df)
    df['ATR'] = df['High'] - df['Low']
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Volatility'] = df['Close'].pct_change().rolling(window=10).std()
    df.dropna(inplace=True)
    return df

# Compute RSI
def compute_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Apply K-Means Clustering for Volatility Classification
def classify_volatility(df):
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Volatility_Cluster'] = kmeans.fit_predict(df[['ATR']])
    
    # Assign labels based on cluster centroids
    cluster_centers = kmeans.cluster_centers_.flatten()
    sorted_indices = np.argsort(cluster_centers)
    cluster_labels = {sorted_indices[0]: 'Low', sorted_indices[1]: 'Medium', sorted_indices[2]: 'High'}
    df['Volatility_Level'] = df['Volatility_Cluster'].map(cluster_labels)
    
    df.dropna(inplace=True)
    return df

# Prepare data for PyTorch
def prepare_lstm_data(df, sequence_length=60):
    features = ['Close', 'MA10', 'MA20', 'RSI14', 'MACD', 'Volatility', 'Volatility_Cluster']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])

    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i+sequence_length])
        y.append(scaled_data[i+sequence_length, 0])  # Predicting 'Close' price

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), scaler

# Define PyTorch LSTM Model
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take the last time step output
        return out

# Train function
def train_model(model, train_loader, optimizer, criterion, epochs=50):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss / len(train_loader):.6f}")

# Main Execution
if __name__ == "__main__":
    df = fetch_stock_data('MSFT')
    df = compute_technical_indicators(df)
    df = classify_volatility(df)

    sequence_length = 60
    X, y, scaler = prepare_lstm_data(df, sequence_length)

    # Split into training and testing sets
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Move to device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StockLSTM(input_size=X.shape[2]).to(device)

    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # DataLoader
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Train Model
    train_model(model, train_loader, optimizer, criterion, epochs=50)

    # Predict Next Day's Price
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test.to(device)).cpu().numpy()
        y_test_np = y_test.numpy()

    # Convert predictions back to real price
    y_pred_real = scaler.inverse_transform(np.column_stack([y_pred, np.zeros((len(y_pred), 6))]))[:, 0]
    y_test_real = scaler.inverse_transform(np.column_stack([y_test_np, np.zeros((len(y_test_np), 6))]))[:, 0]

    # Calculate Accuracy Metrics
    mae = mean_absolute_error(y_test_real, y_pred_real)
    mse = mean_squared_error(y_test_real, y_pred_real)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test_real - y_pred_real) / y_test_real)) * 100
    r2 = r2_score(y_test_real, y_pred_real)

    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")
    print(f"R-squared Score: {r2:.4f}")
    
    # Predict Next Day
    with torch.no_grad():
        last_sequence = X[-1].unsqueeze(0).to(device)
        predicted_price = model(last_sequence).item()
    
    predicted_price = scaler.inverse_transform([[predicted_price, 0, 0, 0, 0, 0, 0]])[0][0]
    print(f"Predicted next day's closing price: {predicted_price:.2f}")
    
    # Display Volatility Classification
    print(df[['Close', 'ATR', 'Volatility_Level']].tail())
