
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import datetime


def custom_kmeans(data, initial_centroids, max_iter=100, tolerance=1e-6):
    """    
    Parameters:
    -----------
    data : numpy.ndarray
        1D array of data points to cluster
    initial_centroids : numpy.ndarray
        Initial centroid values
    max_iter : int
        Maximum number of iterations
    tolerance : float
        Convergence threshold
        
    Returns:
    --------
    centroids : numpy.ndarray
        Final centroid values
    clusters : numpy.ndarray
        Cluster assignments for each data point
    """
    # Reshape data to 1D if needed
    data_1d = data.flatten()
    centroids = initial_centroids.copy()
    
    for iteration in range(max_iter):
        # Calculate distances from each point to each centroid
        distances = np.abs(data_1d.reshape(-1, 1) - centroids.reshape(1, -1))
        
        # Assign points to nearest centroid
        clusters = np.argmin(distances, axis=1)
        
        # Calculate new centroids
        new_centroids = np.array([data_1d[clusters == k].mean() if np.sum(clusters == k) > 0 
                                  else centroids[k] for k in range(len(centroids))])
        
        # Check for convergence
        if np.allclose(new_centroids, centroids, atol=tolerance):
            break
            
        centroids = new_centroids
    
    # Sort centroids in descending order and remap clusters
    centroid_indices = np.argsort(centroids)[::-1]  # Descending order
    sorted_centroids = centroids[centroid_indices]
    
    # Remap cluster labels based on sorted centroids
    remapped_clusters = np.zeros_like(clusters)
    for new_idx, old_idx in enumerate(centroid_indices):
        remapped_clusters[clusters == old_idx] = new_idx
        
    return sorted_centroids, remapped_clusters


def adaptive_supertrend(df, atr_len=10, factor=3.0, training_data_period=100, 
                        highvol=0.75, midvol=0.5, lowvol=0.25,
                        high_multiplier=2.0, mid_multiplier=3.0, low_multiplier=4.0,
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
    high_multiplier : float
        Multiplier for high volatility clusters
    mid_multiplier : float
        Multiplier for medium volatility clusters
    low_multiplier : float
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
            current_multiplier = high_multiplier
        elif cluster == 1:  # Medium volatility cluster
            current_multiplier = mid_multiplier
        else:  # Low volatility cluster
            current_multiplier = low_multiplier
        
        # Calculate SuperTrend with cluster-specific multiplier
        calculate_supertrend_row(df, i, current_multiplier, assigned_centroid, high, low, close)
    
    return df
def calculate_supertrend_row(df, i, factor, atr_value, high='High', low='Low', close='Close'):
    """
    Calculate SuperTrend for a single row (corrected to match Pine Script convention)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with price data
    i : int
        Current row index
    factor : float
        SuperTrend multiplier
    atr_value : float
        Current ATR value or volatility measure
    high, low, close : str
        Column names for High, Low, Close prices
    """
    # Calculate middle price
    hl2 = (df[high].iloc[i] + df[low].iloc[i]) / 2
    
    # Calculate bands
    upper_band = hl2 + factor * atr_value
    lower_band = hl2 - factor * atr_value
    
    # Store raw band values
    df.iloc[i, df.columns.get_loc('upper_band')] = upper_band
    df.iloc[i, df.columns.get_loc('lower_band')] = lower_band
    
    # Initialize with default values if first row
    if i == 0 or pd.isna(df['ADAPT_SUPERTd'].iloc[i-1]):
        # Initialize with default values (Pine Script: -1 for bullish)
        df.iloc[i, df.columns.get_loc('ADAPT_SUPERTd')] = -1  # Default to bullish in Pine
        df.iloc[i, df.columns.get_loc('ADAPT_SUPERT')] = lower_band
        df.iloc[i, df.columns.get_loc('ADAPT_SUPERTl')] = lower_band  # Long band
        df.iloc[i, df.columns.get_loc('ADAPT_SUPERTs')] = upper_band  # Short band
        return
    
    # Get previous values
    prev_trend = df['ADAPT_SUPERT'].iloc[i-1]
    prev_direction = df['ADAPT_SUPERTd'].iloc[i-1]
    prev_upper_band = df['upper_band'].iloc[i-1] if i > 0 and not pd.isna(df['upper_band'].iloc[i-1]) else upper_band
    prev_lower_band = df['lower_band'].iloc[i-1] if i > 0 and not pd.isna(df['lower_band'].iloc[i-1]) else lower_band
    curr_close = df[close].iloc[i]
    
    # Calculate final bands with proper locking mechanism (matching Pine Script)
    final_upper_band = upper_band
    final_lower_band = lower_band
    
    # Implement proper band locking logic (Pine convention: -1 is bullish)
    if prev_direction == -1:  # Previous trend was bullish
        # Lock the lower band in bullish trend (no change to upper band)
        final_lower_band = max(lower_band, prev_lower_band) if curr_close > prev_lower_band else lower_band
        final_upper_band = upper_band
    else:  # Previous trend was bearish (1)
        # Lock the upper band in bearish trend (no change to lower band)
        final_upper_band = min(upper_band, prev_upper_band) if curr_close < prev_upper_band else upper_band
        final_lower_band = lower_band
    
    # Determine new trend direction (Pine convention: -1 is bullish, 1 is bearish)
    if prev_direction == -1:  # Previous was bullish
        if curr_close < prev_trend:
            # Trend changes to bearish
            direction = 1  # Bearish in Pine
            trend_value = final_upper_band
        else:
            # Continue bullish trend
            direction = -1  # Bullish in Pine
            trend_value = final_lower_band
    else:  # Previous was bearish (1)
        if curr_close > prev_trend:
            # Trend changes to bullish
            direction = -1  # Bullish in Pine
            trend_value = final_lower_band
        else:
            # Continue bearish trend
            direction = 1  # Bearish in Pine
            trend_value = final_upper_band
    
    # Store values
    df.iloc[i, df.columns.get_loc('ADAPT_SUPERT')] = trend_value
    df.iloc[i, df.columns.get_loc('ADAPT_SUPERTd')] = direction
    df.iloc[i, df.columns.get_loc('ADAPT_SUPERTl')] = final_lower_band
    df.iloc[i, df.columns.get_loc('ADAPT_SUPERTs')] = final_upper_band


def plot_adaptive_supertrend(stock_name, df, fill_alpha=0.25):
    """
    Plot the Adaptive SuperTrend indicator and ATR with centroids and cluster labels.
    Adjusted for Pine Script convention: -1 = bullish, 1 = bearish
    
    Parameters:
    -----------
    stock_name : str
        Name of the stock for the title
    df : pandas.DataFrame
        DataFrame with calculated Adaptive SuperTrend values
    fill_alpha : float
        Alpha transparency for optional elements
    """
    plot_df = df.dropna(subset=['ADAPT_SUPERT'])
    print("rows =", len(plot_df))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # ====== Subplot 1: Price + SuperTrend + Cluster Markers ======
    ax1.plot(plot_df.index, plot_df['Close'], label='Close', color='black', alpha=0.6)

    # Adaptive SuperTrend line (adjusted for Pine Script convention)
    trend = plot_df['ADAPT_SUPERT'].values
    direction = plot_df['ADAPT_SUPERTd'].values
    dates = mdates.date2num(plot_df.index)

    ax1.plot(dates[:2], trend[:2], color='green' if direction[1] == -1 else 'red',
             linewidth=1.8, label='Bearish')
    ax1.plot(dates[:2], trend[:2], color='red' if direction[1] == -1 else 'green',
             linewidth=1.8, label='Bullish')
    for i in range(2, len(trend)):
        color = 'green' if direction[i] == -1 else 'red'  # -1 is bullish in Pine
        ax1.plot(dates[i-1:i+1], trend[i-1:i+1], color=color, linewidth=1.8)

    # Buy/Sell signals (adjusted for Pine Script convention)
    bullish_cross = (plot_df['ADAPT_SUPERTd'].shift() == 1) & (plot_df['ADAPT_SUPERTd'] == -1)  # to bullish
    bearish_cross = (plot_df['ADAPT_SUPERTd'].shift() == -1) & (plot_df['ADAPT_SUPERTd'] == 1)  # to bearish

    ax1.scatter(plot_df.index[bullish_cross], plot_df['ADAPT_SUPERT'][bullish_cross] * 0.99,
                color='green', marker='^', s=100, label='Buy Signal')
    ax1.scatter(plot_df.index[bearish_cross], plot_df['ADAPT_SUPERT'][bearish_cross] * 1.01,
                color='red', marker='v', s=100, label='Sell Signal')

    # Assigned cluster text on price chart
    cluster_display = {0: ('3', 'red'), 1: ('2', 'orange'), 2: ('1', 'green')}
    for idx, row in plot_df.iterrows():
        cluster = row['cluster']
        if pd.isna(cluster):
            continue
        label, color = cluster_display.get(int(cluster), ('?', 'gray'))
        close_price = row['Close']
        ax1.text(idx, close_price * 1.01, label,
                color=color, fontsize=7, ha='center', va='bottom', alpha=0.8)

    ax1.set_title(f'Adaptive SuperTrend Indicator ({stock_name})', fontsize=16, pad=20)
    ax1.set_ylabel('Price', labelpad=10) 
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', bbox_to_anchor=(0, 1.02))
    ax1.set_ylim(plot_df['Low'].min() * 0.98, plot_df['High'].max() * 1.05)

    # ====== Subplot 2: ATR + Centroids + Cluster Labels ======
    ax2.plot(plot_df.index, plot_df['volatility'], label='Actual ATR', color='purple', alpha=0.7)

    if all(col in plot_df.columns for col in ['low_vol_centroid', 'mid_vol_centroid', 'high_vol_centroid']):
        ax2.plot(plot_df.index, plot_df['low_vol_centroid'], label='Low ATR Cluster', color='green', linestyle='--', alpha=0.7)
        ax2.plot(plot_df.index, plot_df['mid_vol_centroid'], label='Mid ATR Cluster', color='orange', linestyle='--', alpha=0.7)
        ax2.plot(plot_df.index, plot_df['high_vol_centroid'], label='High ATR Cluster', color='red', linestyle='--', alpha=0.7)

    if 'cluster' in plot_df.columns:
        cluster_change = plot_df['cluster'].ne(plot_df['cluster'].shift())

        for idx in plot_df[cluster_change].index:
            cluster = plot_df.loc[idx, 'cluster']
            if pd.isna(cluster):
                continue
            label, color = cluster_display.get(int(cluster), (f"C{int(cluster)}", 'gray'))
            y_pos = plot_df.loc[idx, 'volatility']
            ax2.text(
                idx, y_pos, label,
                color=color,
                fontsize=9, ha='center', va='bottom',
                bbox=dict(facecolor='white', edgecolor=color, alpha=0.6)
            )

    ax2.set_ylabel('ATR / Centroids')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()

    return fig


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


def example():
    stock_name = 'RELIANCE.NS'

    """
    Run an example of the Adaptive SuperTrend indicator on the specified stock
    """
    print(f"Downloading {stock_name} data...")
    # Download sample data
    df = historical_data(stock_name, '1d', 700)
    
    print("Calculating Adaptive SuperTrend...")
    # Calculate Adaptive SuperTrend
    result = adaptive_supertrend(df)
    result.to_excel(f'{stock_name}_adaptive_supertrend.xlsx')
    
    print("Plotting results...")
    # Plot results
    fig = plot_adaptive_supertrend(stock_name, result)
    plt.show()  # This actually displays the plot
    
    # Print some statistics (adjusted for Pine Script convention)
    print("\nSuperTrend Statistics:")
    print(f"Number of bullish trends: {(result['ADAPT_SUPERTd'] == -1).sum()}")  # -1 is bullish
    print(f"Number of bearish trends: {(result['ADAPT_SUPERTd'] == 1).sum()}")   # 1 is bearish
    print(f"Number of trend changes: {(result['ADAPT_SUPERTd'] != result['ADAPT_SUPERTd'].shift()).sum()}")
    
    print("\nVolatility Regime Statistics:")
    print(f"High volatility days: {(result['cluster'] == 0).sum()}")
    print(f"Medium volatility days: {(result['cluster'] == 1).sum()}")
    print(f"Low volatility days: {(result['cluster'] == 2).sum()}")
    
    # Print buy/sell signals
    bullish_cross = (result['ADAPT_SUPERTd'].shift() == 1) & (result['ADAPT_SUPERTd'] == -1)  # to bullish
    bearish_cross = (result['ADAPT_SUPERTd'].shift() == -1) & (result['ADAPT_SUPERTd'] == 1)  # to bearish
    
    print(f"\nNumber of buy signals: {bullish_cross.sum()}")
    print(f"Number of sell signals: {bearish_cross.sum()}")
    
    return result


def get_adaptive_supertrend_json(settings):
    """
    Calculates the Adaptive SuperTrend indicator for a given ticker using supplied settings.
    Returns a JSON string with the following structure:
    
        {
          "success": true/false,
          "error": <error message or null>,
          "data": [
            {
              "Date": <ISO date>,
              "ADAPT_SUPERT": <indicator value>,
              "Close": <close price>,
              "up1": 1 (for upward) or 0 (for downward) or null (if no signal)
            },
            ...
          ]
        }
    
    Parameters:
    -----------
    ticker : str
        Ticker symbol (e.g., 'AAPL', 'RELIANCE.NS').
    settings : dict
        Dictionary with adaptive settings. Expected keys include:
          - atr_len (default 10)
          - factor (default 3.0)
          - training_data_period (default 100)
          - highvol (default 0.75)
          - midvol (default 0.5)
          - lowvol (default 0.25)
          - high_multiplier (default 2.0)
          - mid_multiplier (default 3.0)
          - low_multiplier (default 4.0)
          - days (default 700)
    
    Returns:
    --------
    str
        JSON string containing the data along with success/error fields.
    """

    try:
        ticker = settings["ticker"]
        df = historical_data(ticker, '1d', days=settings.get('days', 700))
        result = adaptive_supertrend(
            df,
            atr_len=settings.get('atr_len', 10),
            factor=settings.get('factor', 3.0),
            training_data_period=settings.get('training_data_period', 100),
            highvol=settings.get('highvol', 0.75),
            midvol=settings.get('midvol', 0.5),
            lowvol=settings.get('lowvol', 0.25),
            high_multiplier=settings.get('high_multiplier', 2.0),
            mid_multiplier=settings.get('mid_multiplier', 3.0),
            low_multiplier=settings.get('low_multiplier', 4.0)
        )
        
        # Compute a single signal field "up1"
        # upward signal: previous ADAPT_SUPERTd == 1 and current ADAPT_SUPERTd == -1 → up1 = 1
        # downward signal: previous ADAPT_SUPERTd == -1 and current ADAPT_SUPERTd == 1 → up1 = 0
        result['up1'] = result['ADAPT_SUPERTd']
        
        result_reset = result.reset_index()
        output_df = result_reset[['Date', 'ADAPT_SUPERT', 'Close', 'up1']]
        data_records = output_df.to_dict(orient='records')
        response = {
            "success": True,
            "error": None,
            "data": data_records
        }
        return json.dumps(response, default=str)
    
    except Exception as e:
        response = {
            "success": False,
            "error": str(e),
            "data": None
        }
        return json.dumps(response)


# Example usage of the new JSON function
if __name__ == "__main__":
    settings = {
        "ticker" : "RELIANCE.NS",
        "atr_len": 10,
        "factor": 3.0,
        "training_data_period": 100,
        "highvol": 0.75,
        "midvol": 0.5,
        "lowvol": 0.25,
        "high_multiplier": 2.0,
        "mid_multiplier": 3.0,
        "low_multiplier": 4.0,
        "days": 700
    }
    json_response = get_adaptive_supertrend_json(settings)
    print(json_response)