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
    df['supertrend'] = np.nan
    df['direction'] = np.nan
    
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
    """Calculate SuperTrend for a single row"""
    hl2 = (df[high].iloc[i] + df[low].iloc[i]) / 2
    
    # Calculate bands
    upper_band = hl2 + factor * atr_value
    lower_band = hl2 - factor * atr_value
    
    # Adjust bands based on previous values (for 'locking' mechanism)
    if i > 0:
        # Explicitly convert to float to ensure scalar values
        prev_supertrend = float(df['supertrend'].iloc[i-1])
        prev_direction = float(df['direction'].iloc[i-1])
        curr_close = float(df[close].iloc[i])
        prev_close = float(df[close].iloc[i-1])
        
        # If no previous data, initialize
        if pd.isna(prev_supertrend) or pd.isna(prev_direction):
            direction = 1  # Default to bullish
            supertrend = lower_band
        else:
            # Logic for SuperTrend calculation
            if prev_direction == 1:  # Previous trend was up
                if curr_close < prev_supertrend:
                    direction = -1  # Switch to bearish
                    supertrend = upper_band
                else:
                    direction = 1
                    supertrend = max(lower_band, prev_supertrend)  # Lock the lowerband
            else:  # Previous trend was down
                if curr_close > prev_supertrend:
                    direction = 1  # Switch to bullish
                    supertrend = lower_band
                else:
                    direction = -1
                    supertrend = min(upper_band, prev_supertrend)  # Lock the upperband
    else:
        # First valid row, initialize
        direction = 1
        supertrend = lower_band
    
    # Store values
    df.iloc[i, df.columns.get_loc('supertrend')] = supertrend
    df.iloc[i, df.columns.get_loc('direction')] = direction


def plot_adaptive_supertrend(df, include_centroids=True):
    """
    Plot the Adaptive SuperTrend indicator results
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with calculated Adaptive SuperTrend values
    include_centroids : bool
        Whether to include centroid values in the plots
    """
    # Filter out NaN values for plotting
    plot_df = df.dropna(subset=['supertrend'])
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    if include_centroids:
        # Create 3 subplots: price/supertrend, volatility/centroids, volatility regime
        gs = plt.GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.1)
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1], sharex=ax1)
        ax3 = plt.subplot(gs[2], sharex=ax1)
    else:
        # Create 2 subplots: price/supertrend and volatility regime
        gs = plt.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.1)
        ax1 = plt.subplot(gs[0])
        ax3 = plt.subplot(gs[1], sharex=ax1)
        ax2 = None
    
    # Plot price
    ax1.plot(plot_df.index, plot_df['Close'], label='Close', color='black', alpha=0.5)
    
    # Plot SuperTrend
    bullish = plot_df['direction'] == 1
    bearish = plot_df['direction'] == -1
    
    ax1.plot(plot_df.index[bullish], plot_df['supertrend'][bullish], 
             color='green', linewidth=1.5, label='SuperTrend (Bullish)')
    ax1.plot(plot_df.index[bearish], plot_df['supertrend'][bearish], 
             color='red', linewidth=1.5, label='SuperTrend (Bearish)')
    
    # Plot trend changes
    trend_changes_bullish = (plot_df['direction'].shift() == -1) & (plot_df['direction'] == 1)
    trend_changes_bearish = (plot_df['direction'].shift() == 1) & (plot_df['direction'] == -1)
    
    ax1.scatter(plot_df.index[trend_changes_bullish], plot_df['supertrend'][trend_changes_bullish], 
                color='green', marker='^', s=100, label='Buy Signal')
    ax1.scatter(plot_df.index[trend_changes_bearish], plot_df['supertrend'][trend_changes_bearish], 
                color='red', marker='v', s=100, label='Sell Signal')
    
    # Plot volatility and centroids
    if include_centroids and ax2 is not None:
        ax2.plot(plot_df.index, plot_df['volatility'], label='ATR', color='purple', alpha=0.7)
        
        # Plot centroids if they exist
        if 'high_vol_centroid' in plot_df.columns:
            ax2.plot(plot_df.index, plot_df['high_vol_centroid'], 
                     label='High Vol', color='red', linestyle='--', alpha=0.7)
            ax2.plot(plot_df.index, plot_df['mid_vol_centroid'], 
                     label='Mid Vol', color='orange', linestyle='--', alpha=0.7)
            ax2.plot(plot_df.index, plot_df['low_vol_centroid'], 
                     label='Low Vol', color='green', linestyle='--', alpha=0.7)
            
        ax2.set_ylabel('ATR / Centroids')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)
    
    # Plot cluster/volatility regime
    if 'cluster' in plot_df.columns:
        # Define colors and labels for volatility levels
        colors = {0: 'red', 1: 'orange', 2: 'green'}
        labels = {0: 'High', 1: 'Medium', 2: 'Low'}
        
        # Create colored rectangles for each cluster value
        for i in range(len(plot_df)-1):
            if pd.isna(plot_df['cluster'].iloc[i]):
                continue
                
            cluster = int(plot_df['cluster'].iloc[i])
            start_date = plot_df.index[i]
            
            # Find next date or end of dataframe
            if i+1 < len(plot_df):
                end_date = plot_df.index[i+1]
            else:
                end_date = start_date + pd.Timedelta(days=1)  # Just add a day for visualization
                
            # Convert dates to numbers for plotting
            start_num = mdates.date2num(start_date)
            end_num = mdates.date2num(end_date)
            
            # Add rectangle patch
            rect = plt.Rectangle((start_num, cluster-0.4), end_num-start_num, 0.8, 
                                color=colors.get(cluster, 'gray'), alpha=0.7)
            ax3.add_patch(rect)
        
        # Set y-ticks and labels for volatility regimes
        ax3.set_ylim(-0.5, 2.5)
        ax3.set_yticks([0, 1, 2])
        ax3.set_yticklabels(['High', 'Medium', 'Low'])
        ax3.set_ylabel('Volatility\nRegime')
        ax3.grid(True, alpha=0.3)
    
    # Customize primary axis
    ax1.set_title('Adaptive SuperTrend Indicator', fontsize=16)
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Improve x-axis date formatting
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    
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
    """
    Run an example of the Adaptive SuperTrend indicator on AAPL stock
    """
    print("Downloading AAPL data...")
    # Download sample data
    df = historical_data('RELIANCE.NS', '1d', 700)
    
    print("Calculating Adaptive SuperTrend...")
    # Calculate Adaptive SuperTrend
    result = adaptive_supertrend(df)
    print(result.columns)
    print("Plotting results...")
    # Plot results
    fig = plot_adaptive_supertrend(result)
    plt.show()  # This actually displays the plot
    
    # Print some statistics
    print("\nSuperTrend Statistics:")
    print(f"Number of bullish trends: {(result['direction'] == 1).sum()}")
    print(f"Number of bearish trends: {(result['direction'] == -1).sum()}")
    print(f"Number of trend changes: {(result['direction'] != result['direction'].shift()).sum()}")
    
    print("\nVolatility Regime Statistics:")
    print(f"High volatility days: {(result['cluster'] == 0).sum()}")
    print(f"Medium volatility days: {(result['cluster'] == 1).sum()}")
    print(f"Low volatility days: {(result['cluster'] == 2).sum()}")
    
    return result


# If the script is run directly, execute the example
if __name__ == "__main__":
    example()
