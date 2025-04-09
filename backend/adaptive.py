import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import datetime


def adaptive_supertrend(df, atr_len=10, factor=3.0, training_data_period=100, 
                        highvol=0.75, midvol=0.5, lowvol=0.25,
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
        SuperTrend factor
    training_data_period : int
        Length of training data for k-means clustering
    highvol : float
        Initial high volatility percentile guess (0-1)
    midvol : float
        Initial medium volatility percentile guess (0-1)
    lowvol : float
        Initial low volatility percentile guess (0-1)
    high, low, close : str
        Column names for High, Low, Close prices
    
    Returns:
    --------
    pandas.DataFrame
        Original dataframe with additional columns for SuperTrend values,
        direction, cluster assignments, and centroids
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
    
    # Initialize SuperTrend columns as requested
    df['ADAPT_SUPERT'] = np.nan    # Trend value
    df['ADAPT_SUPERTd'] = np.nan   # Direction (1 for bullish, -1 for bearish)
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
        
        initial_centroids = np.array([[high_volatility], [medium_volatility], [low_volatility]])
        
        # Perform k-means clustering with 3 clusters and our initial centroids
        kmeans = KMeans(n_clusters=3, init=initial_centroids, n_init=1)
        kmeans.fit(window)
        
        # Get centroids and sort them in descending order (high to low)
        centroids = kmeans.cluster_centers_.flatten()
        centroid_indices = np.argsort(-centroids)  # Descending order
        centroids = centroids[centroid_indices]
        
        high_vol_centroid, mid_vol_centroid, low_vol_centroid = centroids
        
        # Remap labels based on sorted centroids
        remapped_labels = np.zeros_like(kmeans.labels_)
        for new_idx, old_idx in enumerate(centroid_indices):
            remapped_labels[kmeans.labels_ == old_idx] = new_idx
            
        # Count points in each cluster
        high_vol_size = np.sum(remapped_labels == 0)
        mid_vol_size = np.sum(remapped_labels == 1)
        low_vol_size = np.sum(remapped_labels == 2)
        
        # Determine current volatility's cluster
        current_vol = volatility.iloc[i]
        if pd.isna(current_vol):
            continue
            
        distances = [
            abs(current_vol - high_vol_centroid),
            abs(current_vol - mid_vol_centroid),
            abs(current_vol - low_vol_centroid)
        ]
        
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
        
        # Calculate SuperTrend with the assigned centroid
        calculate_supertrend_row(df, i, factor, assigned_centroid, high, low, close)
    
    return df


def calculate_supertrend_row(df, i, factor, atr_value, high='High', low='Low', close='Close'):
    """
    Calculate SuperTrend for a single row
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
        df.iloc[i, df.columns.get_loc('ADAPT_SUPERTd')] = 1  # Default to bullish
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
    
    # Calculate final upper and lower bands with proper locking mechanism
    final_upper_band = upper_band
    final_lower_band = lower_band
    
    # Implement proper band locking logic
    if prev_direction == 1:  # Previous trend was up (bullish)
        # Update upper band (no locking for upper band in bullish trend)
        final_upper_band = upper_band
        
        # Lock lower band if needed
        final_lower_band = max(lower_band, prev_lower_band) if curr_close > prev_lower_band else lower_band
    else:  # Previous trend was down (bearish)
        # Lock upper band if needed
        final_upper_band = min(upper_band, prev_upper_band) if curr_close < prev_upper_band else upper_band
        
        # Update lower band (no locking for lower band in bearish trend)
        final_lower_band = lower_band
    
    # Determine new trend direction
    if prev_direction == 1:  # Previous trend was up
        if curr_close < prev_trend:
            # Trend changes to bearish
            direction = -1
            trend_value = final_upper_band
        else:
            # Continue bullish trend
            direction = 1
            trend_value = final_lower_band
    else:  # Previous trend was down
        if curr_close > prev_trend:
            # Trend changes to bullish
            direction = 1
            trend_value = final_lower_band
        else:
            # Continue bearish trend
            direction = -1
            trend_value = final_upper_band
    
    # Store values
    df.iloc[i, df.columns.get_loc('ADAPT_SUPERT')] = trend_value
    df.iloc[i, df.columns.get_loc('ADAPT_SUPERTd')] = direction

def plot_adaptive_supertrend(stock_name, df, fill_alpha=0.25):
    """
    Plot the Adaptive SuperTrend indicator and ATR with centroids and cluster labels.
    
    Parameters:
    -----------
    stock_name : str
        Name of the stock for the title
    df : pandas.DataFrame
        DataFrame with calculated Adaptive SuperTrend values
    fill_alpha : float
        Alpha transparency for optional elements
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np
    import pandas as pd

    plot_df = df.dropna(subset=['ADAPT_SUPERT'])
    print("rows =", len(plot_df))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # ====== Subplot 1: Price + SuperTrend + Cluster Markers ======
    ax1.plot(plot_df.index, plot_df['Close'], label='Close', color='black', alpha=0.6)

    # Adaptive SuperTrend line
    trend = plot_df['ADAPT_SUPERT'].values
    direction = plot_df['ADAPT_SUPERTd'].values
    dates = mdates.date2num(plot_df.index)

    ax1.plot(dates[:2], trend[:2], color='green' if direction[1] == 1 else 'red',
             linewidth=1.8, label='Bearish')
    ax1.plot(dates[:2], trend[:2], color='red' if direction[1] == 1 else 'green',
             linewidth=1.8, label='Bullish')
    for i in range(2, len(trend)):
        color = 'green' if direction[i] == 1 else 'red'
        ax1.plot(dates[i-1:i+1], trend[i-1:i+1] , color=color, linewidth=1.8)

    # Buy/Sell signals
    bullish_cross = (plot_df['ADAPT_SUPERTd'].shift() == -1) & (plot_df['ADAPT_SUPERTd'] == 1)
    bearish_cross = (plot_df['ADAPT_SUPERTd'].shift() == 1) & (plot_df['ADAPT_SUPERTd'] == -1)

    ax1.scatter(plot_df.index[bullish_cross], plot_df['ADAPT_SUPERT'][bullish_cross] * 0.99,
                color='green', marker='^', s=100, label='Buy Signal')
    ax1.scatter(plot_df.index[bearish_cross], plot_df['ADAPT_SUPERT'][bearish_cross] * 1.01,
                color='red', marker='v', s=100, label='Sell Signal')

    # Assigned cluster text on price chart
    cluster_colors = {1: 'green', 2: 'orange', 3: 'red'}
    cluster_display = {0: ('3', 'red'), 1: ('2', 'orange'), 2: ('1', 'green')}
    for idx, row in plot_df.iterrows():
        cluster = row['cluster']
        label, color = cluster_display.get(cluster, ('?', 'gray'))
        close_price = row['Close']
        ax1.text(idx, close_price * 1.01, label,
                color=color, fontsize=7, ha='center', va='bottom', alpha=0.8)

    ax1.set_title(f'Adaptive SuperTrend Indicator ({stock_name})', fontsize=16)
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
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
            cluster = int(plot_df.loc[idx, 'cluster'])
            label, color = cluster_display.get(cluster, (f"C{cluster}", 'gray'))
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
    Run an example of the Adaptive SuperTrend indicator on AAPL stock
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
    
    # Print some statistics
    print("\nSuperTrend Statistics:")
    print(f"Number of bullish trends: {(result['ADAPT_SUPERTd'] == 1).sum()}")
    print(f"Number of bearish trends: {(result['ADAPT_SUPERTd'] == -1).sum()}")
    print(f"Number of trend changes: {(result['ADAPT_SUPERTd'] != result['ADAPT_SUPERTd'].shift()).sum()}")
    
    print("\nVolatility Regime Statistics:")
    print(f"High volatility days: {(result['cluster'] == 0).sum()}")
    print(f"Medium volatility days: {(result['cluster'] == 1).sum()}")
    print(f"Low volatility days: {(result['cluster'] == 2).sum()}")
    
    return result


# If the script is run directly, execute the example
if __name__ == "__main__":
    example()