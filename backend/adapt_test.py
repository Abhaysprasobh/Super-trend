import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import datetime


def traditional_supertrend(df, atr_len=10, factor=3.0, high='High', low='Low', close='Close'):
    """
    Calculate Traditional SuperTrend with fixed multiplier
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with OHLC data
    atr_len : int
        ATR length for volatility calculation
    factor : float
        SuperTrend multiplier (fixed)
    high, low, close : str
        Column names for High, Low, Close prices
    
    Returns:
    --------
    pandas.DataFrame
        Original dataframe with additional columns for SuperTrend values
    """
    df = df.copy()
    
    # Calculate ATR (Average True Range)
    tr1 = df[high] - df[low]
    tr2 = abs(df[high] - df[close].shift())
    tr3 = abs(df[low] - df[close].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_len).mean()
    
    # Calculate bands
    hl2 = (df[high] + df[low]) / 2
    upper_band = hl2 + factor * atr
    lower_band = hl2 - factor * atr
    
    # Initialize SuperTrend columns
    col_prefix = f'ST_{factor}'
    df[f'{col_prefix}_value'] = np.nan
    df[f'{col_prefix}_d'] = np.nan
    
    # First row
    df.loc[df.index[0], f'{col_prefix}_d'] = -1  # Start as bullish (Pine convention)
    df.loc[df.index[0], f'{col_prefix}_value'] = lower_band.iloc[0]
    
    # Calculate SuperTrend values
    for i in range(1, len(df)):
        # Get previous values
        prev_value = df[f'{col_prefix}_value'].iloc[i-1]
        prev_dir = df[f'{col_prefix}_d'].iloc[i-1]
        curr_close = df[close].iloc[i]
        curr_upper = upper_band.iloc[i]
        curr_lower = lower_band.iloc[i]
        prev_upper = upper_band.iloc[i-1]
        prev_lower = lower_band.iloc[i-1]
        
        # Lock bands
        if prev_dir == -1:  # Bullish
            curr_lower = max(curr_lower, prev_lower) if curr_close > prev_lower else curr_lower
        else:  # Bearish
            curr_upper = min(curr_upper, prev_upper) if curr_close < prev_upper else curr_upper
        
        # Determine direction
        if prev_dir == -1:  # Previous was bullish
            if curr_close < prev_value:
                direction = 1  # Switch to bearish
                value = curr_upper
            else:
                direction = -1  # Stay bullish
                value = curr_lower
        else:  # Previous was bearish
            if curr_close > prev_value:
                direction = -1  # Switch to bullish
                value = curr_lower
            else:
                direction = 1  # Stay bearish
                value = curr_upper
        
        # Store values
        df.loc[df.index[i], f'{col_prefix}_value'] = value
        df.loc[df.index[i], f'{col_prefix}_d'] = direction
    
    return df


def historical_data(symbol, interval, days=700):
    """
    Downloads historical OHLC data using yfinance.
    :param symbol: Ticker symbol (e.g., 'AAPL', 'SPY', etc.)
    :param interval: Data interval (e.g., '1h', '1d')
    :param days: Number of days of historical data to fetch
    :return: DataFrame containing the historical data
    """
    # Download data from yfinance
    data = yf.download(symbol,
                      start=(datetime.datetime.now() - datetime.timedelta(days=days)),
                      end=datetime.datetime.now(),
                      interval=interval,
                      auto_adjust=False,
                      multi_level_index=False)
    data.reset_index(inplace=True)
   
    # Rename "Datetime" to "Date" if it exists
    if 'Datetime' in data.columns:
        data.rename(columns={"Datetime": "Date"}, inplace=True)
    # Otherwise, if there's no "Date" column, you might need to use the first column:
    elif 'Date' not in data.columns:
        data.rename(columns={data.columns[0]: "Date"}, inplace=True)
   
    # Convert Date column to datetime format and set it as index
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date')
   
    return data


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =============================================================================
# Placeholder for custom K-Means clustering function.
def custom_kmeans(data, initial_centroids):
    """
    A simple KMeans clustering function for demonstration.
    
    For each point, assign it to the nearest initial centroid.
    Does not iterate to update centroids.
    
    Parameters:
    -----------
    data : np.array (n_samples, 1)
        Data points to cluster.
    initial_centroids : np.array
        Initial centroid estimates.
        
    Returns:
    --------
    centroids : np.array
        Final centroids (unchanged in this demo).
    labels : np.array
        Cluster labels assigned to each data point.
    """
    centroids = initial_centroids.copy().flatten()
    distances = np.abs(data - centroids.reshape(1, -1))
    labels = np.argmin(distances, axis=1)
    return centroids, labels

# =============================================================================
def adaptive_supertrend(df, atr_len=10, factor=3.0, training_data_period=100, 
                        highvol=0.75, midvol=0.5, lowvol=0.25,
                        high_vol_multiplier=2.0, mid_vol_multiplier=3.0, low_vol_multiplier=4.0,
                        high='High', low='Low', close='Close'):
    """
    Machine Learning Adaptive SuperTrend Indicator 
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with OHLC data
    atr_len : int
        ATR length for volatility calculation
    factor : float
        SuperTrend factor (fallback if cluster-specific multipliers not used)
    training_data_period : int
        Length of training data for custom k-means clustering
    highvol : float
        Initial high volatility percentile guess (0-1)
    midvol : float
        Initial medium volatility percentile guess (0-1)
    lowvol : float
        Initial low volatility percentile guess (0-1)
    high_vol_multiplier : float
        Multiplier for high volatility clusters
    mid_vol_multiplier : float
        Multiplier for medium volatility clusters
    low_vol_multiplier : float
        Multiplier for low volatility clusters
    high, low, close : str
        Column names for High, Low, Close prices
    
    Returns:
    --------
    pandas.DataFrame
        Original dataframe with additional columns for SuperTrend values
    """
    df = df.copy()
    
    # Calculate ATR (Average True Range)
    tr1 = df[high] - df[low]
    tr2 = abs(df[high] - df[close].shift())
    tr3 = abs(df[low] - df[close].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    volatility = tr.rolling(window=atr_len).mean()  # ATR
    
    # Initialize columns
    df['volatility'] = volatility
    df['cluster'] = np.nan
    df['assigned_centroid'] = np.nan
    df['high_vol_centroid'] = np.nan
    df['mid_vol_centroid'] = np.nan
    df['low_vol_centroid'] = np.nan
    df['high_vol_size'] = np.nan
    df['mid_vol_size'] = np.nan
    df['low_vol_size'] = np.nan
    
    # Initialize SuperTrend columns (Pine Script convention)
    df['ADAPT_SUPERT'] = np.nan    # Trend value
    df['ADAPT_SUPERTd'] = np.nan   # Direction (-1 for bullish, 1 for bearish in Pine)
    df['ADAPT_SUPERTl'] = np.nan   # Long band
    df['ADAPT_SUPERTs'] = np.nan   # Short band
    df['upper_band'] = np.nan      # Upper band
    df['lower_band'] = np.nan      # Lower band
    
    # Process data only after we have enough history
    for i in range(training_data_period, len(df)):
        # Get training window
        window = df['volatility'].iloc[i-training_data_period:i].dropna().values.reshape(-1, 1)
        
        if len(window) < training_data_period:
            continue
            
        # Initial centroid estimates based on percentiles
        upper = np.max(window)
        lower = np.min(window)
        
        high_volatility = lower + (upper - lower) * highvol
        medium_volatility = lower + (upper - lower) * midvol
        low_volatility = lower + (upper - lower) * lowvol
        
        initial_centroids = np.array([high_volatility, medium_volatility, low_volatility])
        
        # Use custom K-means clustering that matches Pine Script behavior
        centroids, remapped_labels = custom_kmeans(window, initial_centroids)
        
        high_vol_centroid, mid_vol_centroid, low_vol_centroid = centroids
            
        # Count points in each cluster
        high_vol_size = np.sum(remapped_labels == 0)
        mid_vol_size = np.sum(remapped_labels == 1)
        low_vol_size = np.sum(remapped_labels == 2)
        
        # Determine current volatility's cluster
        current_vol = volatility.iloc[i]
        if pd.isna(current_vol):
            continue
            
        distances = np.abs(current_vol - centroids)
        cluster = np.argmin(distances)
        assigned_centroid = centroids[cluster]
        
        # Store cluster info in dataframe
        df.iloc[i, df.columns.get_loc('cluster')] = cluster
        df.iloc[i, df.columns.get_loc('assigned_centroid')] = assigned_centroid
        df.iloc[i, df.columns.get_loc('high_vol_centroid')] = high_vol_centroid
        df.iloc[i, df.columns.get_loc('mid_vol_centroid')] = mid_vol_centroid
        df.iloc[i, df.columns.get_loc('low_vol_centroid')] = low_vol_centroid
        df.iloc[i, df.columns.get_loc('high_vol_size')] = high_vol_size
        df.iloc[i, df.columns.get_loc('mid_vol_size')] = mid_vol_size
        df.iloc[i, df.columns.get_loc('low_vol_size')] = low_vol_size
        
        # Determine cluster-specific multiplier
        if cluster == 0:  # High volatility cluster
            current_vol_multiplier = high_vol_multiplier
        elif cluster == 1:  # Medium volatility cluster
            current_vol_multiplier = mid_vol_multiplier
        else:  # Low volatility cluster
            current_vol_multiplier = low_vol_multiplier
        
        # Calculate SuperTrend with cluster-specific multiplier
        calculate_supertrend_row(df, i, current_vol_multiplier, assigned_centroid, high, low, close)
    
    return df



# =============================================================================
def calculate_supertrend_row(df, i, factor, atr_value, high='High', low='Low', close='Close'):
    """
    Calculate Adaptive SuperTrend for a given row.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with OHLC and clustering data.
    i : int
        Current row index.
    factor : float
        Cluster-specific multiplier from adaptive_supertrend()
    atr_value : float
        ATR value at the current row.
    high, low, close : str
        Names of the price columns.
    """
    # Compute the middle price
    hl2 = (df[high].iloc[i] + df[low].iloc[i]) / 2
    
    # Calculate bands using the cluster-specific multiplier
    upper_band = hl2 + factor * atr_value
    lower_band = hl2 - factor * atr_value
    
    df.iloc[i, df.columns.get_loc('upper_band')] = upper_band
    df.iloc[i, df.columns.get_loc('lower_band')] = lower_band
    
    # Initialization for the first row or missing previous trend data
    if i == 0 or pd.isna(df['ADAPT_SUPERTd'].iloc[i-1]):
        df.iloc[i, df.columns.get_loc('ADAPT_SUPERTd')] = -1  # default bullish
        df.iloc[i, df.columns.get_loc('ADAPT_SUPERT')] = lower_band
        df.iloc[i, df.columns.get_loc('ADAPT_SUPERTl')] = lower_band
        df.iloc[i, df.columns.get_loc('ADAPT_SUPERTs')] = upper_band
        return
    
    # Retrieve previous SuperTrend values needed for the "band locking" mechanism
    prev_trend = df['ADAPT_SUPERT'].iloc[i-1]
    prev_direction = df['ADAPT_SUPERTd'].iloc[i-1]
    prev_upper_band = df['upper_band'].iloc[i-1] if not pd.isna(df['upper_band'].iloc[i-1]) else upper_band
    prev_lower_band = df['lower_band'].iloc[i-1] if not pd.isna(df['lower_band'].iloc[i-1]) else lower_band
    curr_close = df[close].iloc[i]
    
    # Lock bands based on prior trend:
    final_upper_band = upper_band
    final_lower_band = lower_band
    
    if prev_direction == -1:  # Previous trend was bullish
        final_lower_band = max(lower_band, prev_lower_band) if curr_close > prev_lower_band else lower_band
    else:                     # Previous trend was bearish
        final_upper_band = min(upper_band, prev_upper_band) if curr_close < prev_upper_band else upper_band
    
    # Determine the new trend direction
    if prev_direction == -1:
        if curr_close < prev_trend:
            direction = 1  # switch to bearish
            trend_value = final_upper_band
        else:
            direction = -1  # remain bullish
            trend_value = final_lower_band
    else:
        if curr_close > prev_trend:
            direction = -1  # switch to bullish
            trend_value = final_lower_band
        else:
            direction = 1   # remain bearish
            trend_value = final_upper_band
    
    df.iloc[i, df.columns.get_loc('ADAPT_SUPERT')] = trend_value
    df.iloc[i, df.columns.get_loc('ADAPT_SUPERTd')] = direction
    df.iloc[i, df.columns.get_loc('ADAPT_SUPERTl')] = final_lower_band
    df.iloc[i, df.columns.get_loc('ADAPT_SUPERTs')] = final_upper_band




# =============================================================================
def generate_signals(df, supertrend_col, direction_col):
    """
    Generate buy and sell signals based on changes in the SuperTrend direction.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with SuperTrend values.
    supertrend_col : str
        Column name for the SuperTrend value.
    direction_col : str
        Column name for the SuperTrend direction.
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with additional columns for signals.
    """
    df = df.copy()
    signal_col = f"{supertrend_col}_signal"
    df[signal_col] = df[direction_col].diff()
    
    # Buy signal: change from bearish to bullish (-2 diff)
    df[f"{supertrend_col}_buy"] = np.where(df[signal_col] == -2, 1, 0)
    # Sell signal: change from bullish to bearish (2 diff)
    df[f"{supertrend_col}_sell"] = np.where(df[signal_col] == 2, 1, 0)
    
    return df

# =============================================================================
def backtest_supertrend(df, supertrend_col, buy_col, sell_col, close='Close', initial_capital=100000):
    """
    Perform a simple backtest based on the SuperTrend buy and sell signals.
    """
    df = df.copy()
    portfolio_col = f'{supertrend_col}_portfolio'  # Unique column name per strategy
    df[portfolio_col] = float(initial_capital)  # Initialize as float
    
    in_position = False
    shares = 0
    cash = float(initial_capital)
    
    for i in range(len(df)):
        if df[buy_col].iloc[i] == 1 and not in_position:
            entry_price = df[close].iloc[i]
            shares = cash / entry_price
            cash = 0.0
            in_position = True
        elif df[sell_col].iloc[i] == 1 and in_position:
            exit_price = df[close].iloc[i]
            cash = shares * exit_price
            shares = 0.0
            in_position = False
        
        # Update portfolio value
        df.loc[df.index[i], portfolio_col] = cash + shares * df[close].iloc[i]
    
    # Calculate metrics
    initial_price = df[close].iloc[0]
    final_price = df[close].iloc[-1]
    strategy_return = (df[portfolio_col].iloc[-1] / initial_capital - 1) * 100
    
    return df, {
        'final_capital': df[portfolio_col].iloc[-1],
        'total_return': strategy_return
    }

# =============================================================================
def plot_supertrend_with_signals(stock_name, df, traditional_factor=3.0):
    """
    Plot the comparison of Adaptive and Traditional SuperTrend signals.
    
    Parameters:
    -----------
    stock_name : str
        Stock ticker or name.
    df : pandas.DataFrame
        DataFrame containing price and indicator data.
    traditional_factor : float
        The multiplier used in the traditional SuperTrend.
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated plot.
    """
    plot_df = df.dropna(subset=['ADAPT_SUPERT']).copy()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12),
                                   gridspec_kw={'height_ratios': [3, 1]},
                                   sharex=True)
    
    ax1.plot(plot_df.index, plot_df['Close'], label='Close Price', color='black', alpha=0.8)
    # Assuming traditional_supertrend has created columns using this naming convention:
    trad_prefix = f'ST_{traditional_factor}'
    ax1.plot(plot_df.index, plot_df[f'{trad_prefix}_value'], label=f'Traditional ST {traditional_factor}x', color='blue', alpha=0.7)
    ax1.plot(plot_df.index, plot_df['ADAPT_SUPERT'], label='Adaptive ST', color='red', alpha=0.7)
    
    ax1.scatter(plot_df[plot_df[f'{trad_prefix}_buy'] == 1].index,
                plot_df.loc[plot_df[f'{trad_prefix}_buy'] == 1, 'Close'] * 0.99,
                marker='^', color='green', s=100, label='Traditional ST Buy')
    ax1.scatter(plot_df[plot_df['ADAPT_SUPERT_buy'] == 1].index,
                plot_df.loc[plot_df['ADAPT_SUPERT_buy'] == 1, 'Close'] * 0.98,
                marker='^', color='darkgreen', s=120, label='Adaptive ST Buy')
    
    ax1.scatter(plot_df[plot_df[f'{trad_prefix}_sell'] == 1].index,
                plot_df.loc[plot_df[f'{trad_prefix}_sell'] == 1, 'Close'] * 1.01,
                marker='v', color='red', s=100, label='Traditional ST Sell')
    ax1.scatter(plot_df[plot_df['ADAPT_SUPERT_sell'] == 1].index,
                plot_df.loc[plot_df['ADAPT_SUPERT_sell'] == 1, 'Close'] * 1.02,
                marker='v', color='darkred', s=120, label='Adaptive ST Sell')
    
    ax1.set_title(f"SuperTrend Comparison for {stock_name}", fontsize=16)
    ax1.set_ylabel("Price", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    ax2.plot(plot_df.index, plot_df[f'{trad_prefix}_portfolio'], label='Traditional ST Portfolio', color='blue')
    ax2.plot(plot_df.index, plot_df['ADAPT_SUPERT_portfolio'], label='Adaptive ST Portfolio', color='red')
    ax2.set_title("Portfolio Performance", fontsize=14)
    ax2.set_ylabel("Portfolio Value", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    
    return fig

import json

def supertrend_strategy_comparison_json(ticker, days=700, 
                                        high_vol_multiplier=3, 
                                        mid_vol_multiplier=2, 
                                        low_vol_multiplier=1):
    df = historical_data(ticker, '1d', days)
    
    # Traditional SuperTrend
    df = traditional_supertrend(df, factor=high_vol_multiplier)
    trad_prefix = f'ST_{high_vol_multiplier}'
    df = generate_signals(df, trad_prefix, f'{trad_prefix}_d')
    df, trad_metrics = backtest_supertrend(df, trad_prefix, f'{trad_prefix}_buy', f'{trad_prefix}_sell')
    
    # Adaptive SuperTrend
    df = adaptive_supertrend(df,
                             high_vol_multiplier=high_vol_multiplier,
                             mid_vol_multiplier=mid_vol_multiplier,
                             low_vol_multiplier=low_vol_multiplier)
    df = generate_signals(df, 'ADAPT_SUPERT', 'ADAPT_SUPERTd')
    df, adapt_metrics = backtest_supertrend(df, 'ADAPT_SUPERT', 'ADAPT_SUPERT_buy', 'ADAPT_SUPERT_sell')
    
    # Prepare signals
    trad_signals = df[['Close', f'{trad_prefix}_buy', f'{trad_prefix}_sell']].copy()
    trad_signals.columns = ['close', 'buy', 'sell']
    trad_signals['date'] = df.index.astype(str)
    
    adapt_signals = df[['Close', 'ADAPT_SUPERT_buy', 'ADAPT_SUPERT_sell']].copy()
    adapt_signals.columns = ['close', 'buy', 'sell']
    adapt_signals['date'] = df.index.astype(str)
    
    # JSON-friendly conversion
    trad_list = trad_signals.to_dict(orient='records')
    adapt_list = adapt_signals.to_dict(orient='records')
    
    better = "Adaptive SuperTrend" if adapt_metrics['final_capital'] > trad_metrics['final_capital'] else "Traditional SuperTrend"
    
    result = {
        "perf": {
            "super": round(trad_metrics['final_capital'], 2),
            "adapt": round(adapt_metrics['final_capital'], 2), 
            "best": better
        },
        "super": trad_list,
        "adapt": adapt_list
    }

    return  json.dumps(result)



import json

if __name__ == '__main__':
    output = supertrend_strategy_comparison_json(
        ticker='RELIANCE.NS',
        days=700,
        high_vol_multiplier=3,
        mid_vol_multiplier=2,
        low_vol_multiplier=1
    )

    # If needed, convert to actual JSON string
    print(output)
