import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

def fetch_stock_data(ticker, period="6mo"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    if df.empty:
        raise ValueError("No stock data fetched. Check ticker symbol or internet connection.")
    return df

def compute_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def adaptive_supertrend(df, base_atr_period=14, base_multiplier=3):
    df['ATR'] = df['Close'].rolling(base_atr_period).std()
    df['EMA_ATR'] = df['ATR'].ewm(span=10, adjust=False).mean()
    
    df['Volatility_Score'] = df['ATR'] / df['EMA_ATR']
    df['Adaptive_Multiplier'] = base_multiplier * df['Volatility_Score']
    df['Upper_Band'] = df['Close'] + (df['Adaptive_Multiplier'] * df['ATR'])
    df['Lower_Band'] = df['Close'] - (df['Adaptive_Multiplier'] * df['ATR'])
    
    df['Supertrend'] = np.nan
    trend = 1  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Upper_Band'].iloc[i-1]:
            trend = 1
        elif df['Close'].iloc[i] < df['Lower_Band'].iloc[i-1]:
            trend = -1
        
        df.loc[df.index[i], 'Supertrend'] = trend
    
    return df

def feature_engineering(df):
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['EMA10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['RSI14'] = compute_rsi(df)
    df = adaptive_supertrend(df)
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility_10'] = df['Daily_Return'].rolling(window=10).std()
    df['Volatility_20'] = df['Daily_Return'].rolling(window=20).std()
    df['Annualized_Volatility'] = df['Volatility_20'] * np.sqrt(252)
    df['Min_Volatility'] = df['Volatility_20'].rolling(window=20).min()
    df['Max_Volatility'] = df['Volatility_20'].rolling(window=20).max()
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    return df.dropna()

def prepare_data(df):
    features = ['MA5', 'MA10', 'MA20', 'EMA10', 'EMA20', 'RSI14', 'Supertrend', 'Volatility_10', 'Volatility_20', 'Annualized_Volatility', 'Min_Volatility', 'Max_Volatility']
    X = df[features]
    y = df['Target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    return train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

def train_models(X_train, X_test, y_train, y_test):
    param_grid = {'n_neighbors': np.arange(1, 50, 2)}
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_k = grid_search.best_params_['n_neighbors']
    
    knn = KNeighborsClassifier(n_neighbors=best_k)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    
    models = {'KNN': knn, 'RandomForest': rf, 'XGBoost': xgb}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"{name} Model Accuracy: {accuracy_score(y_test, y_pred)}")
        print(classification_report(y_test, y_pred))
    
    return models

def predict_next_day(models, X_scaled):
    latest_data = X_scaled[-1].reshape(1, -1)
    predictions = {name: model.predict(latest_data)[0] for name, model in models.items()}
    for name, pred in predictions.items():
        print(f"Predicted movement ({name}): {'Up' if pred == 1 else 'Down'}")

def plot_volatility(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Volatility_20'], label="20-Day Volatility", color="blue", alpha=0.6)
    plt.plot(df.index, df['Min_Volatility'], label="Min Volatility (20-Day)", color="green", linestyle="dashed")
    plt.plot(df.index, df['Max_Volatility'], label="Max Volatility (20-Day)", color="red", linestyle="dashed")
    plt.title("Stock Volatility Representation (Min & Max)")
    plt.legend()
    plt.show()

# Main Execution
if __name__ == "__main__":
    df = fetch_stock_data('MSFT')
    df = feature_engineering(df)
    X_train, X_test, y_train, y_test = prepare_data(df)
    models = train_models(X_train, X_test, y_train, y_test)
    predict_next_day(models, X_test)
    plot_volatility(df)
