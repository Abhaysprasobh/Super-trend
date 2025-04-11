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
    df['atr'] = atr  # Store ATR for reference
    
    # Calculate bands
    hl2 = (df[high] + df[low]) / 2
    upper_band = hl2 + factor * atr
    lower_band = hl2 - factor * atr
    
    # Initialize SuperTrend columns
    col_prefix = f'ST_{factor}'
    df[f'{col_prefix}_value'] = np.nan
    df[f'{col_prefix}_d'] = np.nan
    
    # Find first valid ATR index to avoid NaN contamination
    first_valid_idx = atr.first_valid_index()
    if first_valid_idx is None:
        raise ValueError("Not enough data to calculate ATR")
    
    # Get integer location of first valid index
    start_idx = df.index.get_loc(first_valid_idx)
    
    # Initialize at first valid index
    df.loc[df.index[start_idx], f'{col_prefix}_d'] = -1  # Start as bullish (Pine convention)
    df.loc[df.index[start_idx], f'{col_prefix}_value'] = lower_band.iloc[start_idx]
    
    # Calculate SuperTrend values
    for i in range(start_idx + 1, len(df)):
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
    
    # Process data only after we have enough history for both ATR and training
    first_valid_volatility = volatility.first_valid_index()
    if first_valid_volatility is None:
        raise ValueError("Not enough data to calculate ATR")
    
    # Get minimum start index considering both ATR warmup and training period
    start_idx = max(
        df.index.get_loc(first_valid_volatility),
        training_data_period
    )
    
    # Process data only after we have enough history
    for i in range(start_idx, len(df)):
        # Get training window
        window = df['volatility'].iloc[i-training_data_period:i].dropna().values
        
        if len(window) < training_data_period * 0.9:  # Allow some missing values but ensure sufficient data
            continue
            
        # Initial centroid estimates based on percentiles
        upper = np.max(window)
        lower = np.min(window)
        
        # Calculate initial centroids based on percentiles
        high_volatility = lower + (upper - lower) * highvol
        medium_volatility = lower + (upper - lower) * midvol
        low_volatility = lower + (upper - lower) * lowvol
        
        initial_centroids = np.array([high_volatility, medium_volatility, low_volatility])
        
        # Use custom K-means clustering that matches Pine Script behavior
        centroids, remapped_labels = custom_kmeans(window, initial_centroids)
        
        # After sorting, cluster 0 = highest volatility, 2 = lowest
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
        # Cluster 0 = highest volatility, 1 = medium volatility, 2 = lowest volatility
        if cluster == 0:  # High volatility cluster
            current_vol_multiplier = high_vol_multiplier
        elif cluster == 1:  # Medium volatility cluster
            current_vol_multiplier = mid_vol_multiplier
        else:  # Low volatility cluster
            current_vol_multiplier = low_vol_multiplier
        
        # Calculate SuperTrend with cluster-specific multiplier
        calculate_supertrend_row(df, i, current_vol_multiplier, assigned_centroid, high, low, close)
    
    return df


def generate_signals(df, supertrend_col, direction_col):
    """
    Generate buy and sell signals based on changes in the SuperTrend direction.
    This properly handles the signals for both traditional and adaptive SuperTrend.
    
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
    
    # Add columns for signals (initialize with 0)
    df[f"{supertrend_col}_buy"] = 0
    df[f"{supertrend_col}_sell"] = 0
    
    # The first row cannot generate signals
    for i in range(1, len(df)):
        prev_direction = df[direction_col].iloc[i-1]
        curr_direction = df[direction_col].iloc[i]
        
        # Check if direction changed
        if pd.notna(prev_direction) and pd.notna(curr_direction):
            # Direction changed from bearish (1) to bullish (-1)
            if prev_direction == 1 and curr_direction == -1:
                df.iloc[i, df.columns.get_loc(f"{supertrend_col}_buy")] = 1
            
            # Direction changed from bullish (-1) to bearish (1)
            elif prev_direction == -1 and curr_direction == 1:
                df.iloc[i, df.columns.get_loc(f"{supertrend_col}_sell")] = 1
    
    return df


def backtest_supertrend(df, supertrend_col, buy_col, sell_col, close='Close', initial_capital=100000, transaction_cost_pct=0.1):
    """
    Perform a simple backtest based on the SuperTrend buy and sell signals with transaction costs.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with price and signal data.
    supertrend_col : str
        Base column name for the SuperTrend strategy (used for naming the portfolio column).
    buy_col : str
        Column name for buy signals.
    sell_col : str
        Column name for sell signals.
    close : str
        Column name for closing prices.
    initial_capital : float
        Initial capital for the backtest.
    transaction_cost_pct : float
        Transaction cost percentage (e.g., 0.1 for 0.1%).
        
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with added portfolio value column.
    metrics : dict
        Dictionary with backtest metrics.
    """
    df = df.copy()
    portfolio_col = f'{supertrend_col}_portfolio'  # Unique column name per strategy
    df[portfolio_col] = float(initial_capital)  # Initialize as float
    
    in_position = False
    shares = 0
    cash = float(initial_capital)
    entry_price = 0
    total_trades = 0
    winning_trades = 0
    losing_trades = 0
    
    for i in range(len(df)):
        current_price = df[close].iloc[i]
        
        if df[buy_col].iloc[i] == 1 and not in_position:
            # FIXED: Calculate maximum shares considering transaction cost
            entry_price = current_price
            total_cost_per_share = entry_price * (1 + transaction_cost_pct/100)
            shares = cash / total_cost_per_share  # Uses all available cash
            transaction_cost = shares * entry_price * transaction_cost_pct/100
            
            cash = 0.0
            in_position = True
            total_trades += 1
            
        elif df[sell_col].iloc[i] == 1 and in_position:
            exit_price = current_price
            transaction_cost = shares * exit_price * transaction_cost_pct/100
            cash = shares * exit_price - transaction_cost
            
            # Track trade performance
            if exit_price > entry_price:
                winning_trades += 1
            elif exit_price < entry_price:
                losing_trades += 1
                
            shares = 0.0
            in_position = False
        
        # Update portfolio value
        df.loc[df.index[i], portfolio_col] = cash + shares * current_price
    
    # Calculate metrics
    initial_price = df[close].iloc[0]
    final_price = df[close].iloc[-1]
    buy_hold_return = (final_price / initial_price - 1) * 100
    strategy_return = (df[portfolio_col].iloc[-1] / initial_capital - 1) * 100
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
    
    return df, {
        'final_capital': df[portfolio_col].iloc[-1],
        'total_return': strategy_return,
        'buy_hold_return': buy_hold_return,
        'total_trades': total_trades,
        'win_rate': win_rate
    }


def supertrend_strategy_comparison(ticker, days=700, 
                                   high_vol_multiplier=3, 
                                   mid_vol_multiplier=2, 
                                   low_vol_multiplier=1,
                                   transaction_cost_pct=0.1,
                                   reversed_signals=False):
    """
    Compare Traditional and Adaptive SuperTrend strategies with correctly generated signals.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol.
    days : int
        Number of historical days to analyze.
    high_vol_multiplier : float
        Multiplier for high volatility periods.
    mid_vol_multiplier : float
        Multiplier for medium volatility periods.
    low_vol_multiplier : float
        Multiplier for low volatility periods.
    transaction_cost_pct : float
        Transaction cost percentage.
    reversed_signals : bool
        If True, buy on bearish trend start and sell on bullish trend start.
        If False, buy on bullish trend start and sell on bearish trend start (standard).
        
    Returns:
    --------
    fig : matplotlib Figure
        Plot comparing the two strategies
    df : pandas DataFrame
        Data used for the plot
    """
    # Get historical data
    df = historical_data(ticker, '1d', days)
    
    # Traditional SuperTrend
    df = traditional_supertrend(df, factor=high_vol_multiplier)
    trad_prefix = f'ST_{high_vol_multiplier}'
    df = generate_signals(df, trad_prefix, f'{trad_prefix}_d')
    
    # Adaptive SuperTrend
    df = adaptive_supertrend(df,
                            high_vol_multiplier=high_vol_multiplier,
                            mid_vol_multiplier=mid_vol_multiplier,
                            low_vol_multiplier=low_vol_multiplier)
    df = generate_signals(df, 'ADAPT_SUPERT', 'ADAPT_SUPERTd')
    
    # Reverse signals if requested (opposite of standard trend-following)
    if reversed_signals:
        # Swap buy and sell signals
        for prefix in [trad_prefix, 'ADAPT_SUPERT']:
            buy_col = f"{prefix}_buy"
            sell_col = f"{prefix}_sell"
            temp = df[buy_col].copy()
            df[buy_col] = df[sell_col]
            df[sell_col] = temp
    
    # Run backtest for both strategies
    df, trad_metrics = backtest_supertrend(df, trad_prefix, 
                                           f'{trad_prefix}_buy', 
                                           f'{trad_prefix}_sell',
                                           transaction_cost_pct=transaction_cost_pct)
    
    df, adapt_metrics = backtest_supertrend(df, 'ADAPT_SUPERT', 
                                            'ADAPT_SUPERT_buy', 
                                            'ADAPT_SUPERT_sell',
                                            transaction_cost_pct=transaction_cost_pct)
    
    # Create plot to visualize results
    plot_df = df.dropna(subset=['ADAPT_SUPERT']).copy()
    
    fig = plt.figure(figsize=(16, 20))
    
    # Define GridSpec for layout
    gs = plt.GridSpec(5, 4, figure=fig)
    
    # Main price chart with SuperTrend lines
    ax1 = fig.add_subplot(gs[0:2, :])
    ax1.plot(plot_df.index, plot_df['Close'], label='Close Price', color='black', alpha=0.8)
    ax1.plot(plot_df.index, plot_df[f'{trad_prefix}_value'], label=f'Traditional ST {high_vol_multiplier}x', 
             color='blue', alpha=0.7)
    ax1.plot(plot_df.index, plot_df['ADAPT_SUPERT'], label='Adaptive ST', color='red', alpha=0.7)
    
    # Buy signals
    buy_trad = plot_df[plot_df[f'{trad_prefix}_buy'] == 1]
    buy_adapt = plot_df[plot_df['ADAPT_SUPERT_buy'] == 1]
    
    if not buy_trad.empty:
        ax1.scatter(buy_trad.index, buy_trad['Close'] * 0.99,
                   marker='^', color='green', s=100, label='Traditional ST Buy')
    
    if not buy_adapt.empty:
        ax1.scatter(buy_adapt.index, buy_adapt['Close'] * 0.98,
                   marker='^', color='darkgreen', s=120, label='Adaptive ST Buy')
    
    # Sell signals
    sell_trad = plot_df[plot_df[f'{trad_prefix}_sell'] == 1]
    sell_adapt = plot_df[plot_df['ADAPT_SUPERT_sell'] == 1]
    
    if not sell_trad.empty:
        ax1.scatter(sell_trad.index, sell_trad['Close'] * 1.01,
                   marker='v', color='red', s=100, label='Traditional ST Sell')
    
    if not sell_adapt.empty:
        ax1.scatter(sell_adapt.index, sell_adapt['Close'] * 1.02,
                   marker='v', color='darkred', s=120, label='Adaptive ST Sell')
    
    signal_mode = "Reversed (Buy on Bearish, Sell on Bullish)" if reversed_signals else "Standard (Buy on Bullish, Sell on Bearish)"
    ax1.set_title(f"SuperTrend Comparison for {ticker} - {signal_mode}", fontsize=16)
    ax1.set_ylabel("Price", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Portfolio performance
    ax2 = fig.add_subplot(gs[2, :])
    ax2.plot(plot_df.index, plot_df[f'{trad_prefix}_portfolio'], label='Traditional ST Portfolio', color='blue')
    ax2.plot(plot_df.index, plot_df['ADAPT_SUPERT_portfolio'], label='Adaptive ST Portfolio', color='red')
    
    # Add buy-hold portfolio for comparison
    if 'buy_hold' not in plot_df.columns:
        initial_capital = 100000
        plot_df['buy_hold'] = initial_capital * (plot_df['Close'] / plot_df['Close'].iloc[0])
    ax2.plot(plot_df.index, plot_df['buy_hold'], label='Buy & Hold', color='purple', linestyle='--')
    
    ax2.set_title("Portfolio Performance", fontsize=14)
    ax2.set_ylabel("Portfolio Value ($)", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    # Volatility clusters
    ax3 = fig.add_subplot(gs[3, :])
    colors = {0: 'red', 1: 'orange', 2: 'green'}
    cluster_labels = {0: 'High Volatility', 1: 'Medium Volatility', 2: 'Low Volatility'}
    
    # Plot the volatility line
    ax3.plot(plot_df.index, plot_df['volatility'], color='gray', alpha=0.4, label='ATR Volatility')
    
    # Plot the centroids as horizontal lines
    valid_centroids = plot_df[['high_vol_centroid', 'mid_vol_centroid', 'low_vol_centroid']].iloc[-1]
    for i, (centroid_name, centroid_value) in enumerate(valid_centroids.items()):
        if pd.notna(centroid_value):
            ax3.axhline(y=centroid_value, color=colors[i], linestyle='--', alpha=0.7,
                       label=f"{cluster_labels[i]} Centroid: {centroid_value:.5f}")
    
    # Color the points by cluster
    for cluster in [0, 1, 2]:
        cluster_df = plot_df[plot_df['cluster'] == cluster].dropna(subset=['volatility'])
        if not cluster_df.empty:
            ax3.scatter(cluster_df.index, cluster_df['volatility'], 
                       color=colors.get(cluster, 'gray'), alpha=0.7, 
                       label=f'Cluster {cluster}: {cluster_labels[cluster]}')
    
    ax3.set_title("Volatility Clusters", fontsize=14)
    ax3.set_ylabel("ATR Volatility", fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left')
    
    # Direction and Multiplier chart
    ax4 = fig.add_subplot(gs[4, :])
    
    # Plot direction
    ax4.plot(plot_df.index, plot_df[f'{trad_prefix}_d'], 
             label='Traditional Direction', color='blue', drawstyle='steps-post', alpha=0.7)
    ax4.plot(plot_df.index, plot_df['ADAPT_SUPERTd'], 
             label='Adaptive Direction', color='red', drawstyle='steps-post', alpha=0.7)
    
    # Set y-axis limits to show direction clearly
    ax4.set_ylim(-1.5, 1.5)
    
    # Add reference line at y=0
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.2)
    
    ax4.set_title("SuperTrend Direction (-1: Bullish, 1: Bearish)", fontsize=14)
    ax4.set_ylabel("Direction", fontsize=12)
    ax4.set_yticks([-1, 1])
    ax4.set_yticklabels(['Bullish (-1)', 'Bearish (1)'])
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper left')
    
    # Format x-axis dates
    for ax in [ax1, ax2, ax3, ax4]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Add text box with performance metrics
    perf_text = (
        f"Traditional SuperTrend ({high_vol_multiplier}x):\n"
        f"  Final Capital: ${trad_metrics['final_capital']:.2f}\n"
        f"  Total Return: {trad_metrics['total_return']:.2f}%\n"
        f"  Win Rate: {trad_metrics['win_rate']:.2f}%\n"
        f"  Total Trades: {trad_metrics['total_trades']}\n\n"
        f"Adaptive SuperTrend:\n"
        f"  Final Capital: ${adapt_metrics['final_capital']:.2f}\n"
        f"  Total Return: {adapt_metrics['total_return']:.2f}%\n"
        f"  Win Rate: {adapt_metrics['win_rate']:.2f}%\n"
        f"  Total Trades: {adapt_metrics['total_trades']}\n\n"
        f"Buy & Hold Return: {trad_metrics['buy_hold_return']:.2f}%\n"
        f"Transaction Cost: {transaction_cost_pct}%\n"
        f"Signal Mode: {signal_mode}\n"
        f"Volatility Multipliers: {high_vol_multiplier}x (high), {mid_vol_multiplier}x (mid), {low_vol_multiplier}x (low)"
    )
    
    fig.text(0.05, 0.01, perf_text, fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for text box
    
    print(f"Traditional SuperTrend Signals: {buy_trad.shape[0]} buy, {sell_trad.shape[0]} sell")
    print(f"Adaptive SuperTrend Signals: {buy_adapt.shape[0]} buy, {sell_adapt.shape[0]} sell")
    plt.show()
    return fig, df



def get_supertrend_strategy_data(ticker, days=700, 
                                 high_vol_multiplier=3, 
                                 mid_vol_multiplier=2, 
                                 low_vol_multiplier=1,
                                 transaction_cost_pct=0.1,
                                 reversed_signals=False):
    """
    Generate JSON data with performance metrics and trade points for both Adaptive
    and Traditional SuperTrend strategies.
    """
    try:
        df = historical_data(ticker, '1d', days)

        df = traditional_supertrend(df, factor=high_vol_multiplier)
        trad_prefix = f'ST_{high_vol_multiplier}'
        df = generate_signals(df, trad_prefix, f'{trad_prefix}_d')

        df = adaptive_supertrend(df,
                                 high_vol_multiplier=high_vol_multiplier,
                                 mid_vol_multiplier=mid_vol_multiplier,
                                 low_vol_multiplier=low_vol_multiplier)
        df = generate_signals(df, 'ADAPT_SUPERT', 'ADAPT_SUPERTd')

        if reversed_signals:
            for prefix in [trad_prefix, 'ADAPT_SUPERT']:
                buy_col = f"{prefix}_buy"
                sell_col = f"{prefix}_sell"
                df[buy_col], df[sell_col] = df[sell_col].copy(), df[buy_col].copy()

        df, trad_metrics = backtest_supertrend(df, trad_prefix, 
                                               f'{trad_prefix}_buy', 
                                               f'{trad_prefix}_sell',
                                               transaction_cost_pct=transaction_cost_pct)
        df, adapt_metrics = backtest_supertrend(df, 'ADAPT_SUPERT', 
                                                'ADAPT_SUPERT_buy', 
                                                'ADAPT_SUPERT_sell',
                                                transaction_cost_pct=transaction_cost_pct)

        initial_close = df['Close'].iloc[0]
        final_close = df['Close'].iloc[-1]
        buy_hold_return = ((final_close / initial_close) - 1) * 100

        result = {
            'ticker': ticker,
            'performance': {
                'Supertrend': {
                    'final_capital': float(trad_metrics['final_capital']),
                    'total_return': float(trad_metrics['total_return']),
                    'total_trades': int(trad_metrics['total_trades']),
                    'profitable_trades': int(trad_metrics['win_rate'] * trad_metrics['total_trades'] / 100)
                },
                'adaptive': {
                    'final_capital': float(adapt_metrics['final_capital']),
                    'total_return': float(adapt_metrics['total_return']),
                    'total_trades': int(adapt_metrics['total_trades']),
                    'profitable_trades': int(adapt_metrics['win_rate'] * adapt_metrics['total_trades'] / 100)
                },
                'buy_hold_return': float(buy_hold_return)
            },
            'adapt': [],
            'super': [],
            'params': {
                'days': days,
                'high_vol_multiplier': high_vol_multiplier,
                'mid_vol_multiplier': mid_vol_multiplier,
                'low_vol_multiplier': low_vol_multiplier,
                'transaction_cost_pct': transaction_cost_pct,
                'reversed_signals': reversed_signals
            }
        }

        for idx, row in df.iterrows():
            date_str = idx.strftime('%Y-%m-%d')
            if pd.notna(row['ADAPT_SUPERT']):
                result['adapt'].append({
                    'date': date_str,
                    'close': float(row['Close']),
                    'trend_value': float(row['ADAPT_SUPERT']),
                    'direction': int(row['ADAPT_SUPERTd']),
                    'buy_signal': int(row['ADAPT_SUPERT_buy']),
                    'sell_signal': int(row['ADAPT_SUPERT_sell']),
                    'portfolio_value': float(row['ADAPT_SUPERT_portfolio'])
                })
            if pd.notna(row[f'{trad_prefix}_value']):
                result['super'].append({
                    'date': date_str,
                    'close': float(row['Close']),
                    'trend_value': float(row[f'{trad_prefix}_value']),
                    'direction': int(row[f'{trad_prefix}_d']),
                    'buy_signal': int(row[f'{trad_prefix}_buy']),
                    'sell_signal': int(row[f'{trad_prefix}_sell']),
                    'portfolio_value': float(row[f'{trad_prefix}_portfolio'])
                })

        result['summary'] = {
            'start_date': result['adapt'][0]['date'] if result['adapt'] else None,
            'end_date': result['adapt'][-1]['date'] if result['adapt'] else None,
            'days_analyzed': len(result['adapt']),
            'total_supertrend_signals': sum(item['buy_signal'] + item['sell_signal'] for item in result['super']),
            'total_adaptive_signals': sum(item['buy_signal'] + item['sell_signal'] for item in result['adapt']),
        }
        print(f"Adaptive signals: {sum(d['buy_signal'] for d in result['adapt'])} buys, {sum(d['sell_signal'] for d in result['adapt'])} sells")
        print(f"Supertrend signals: {sum(d['buy_signal'] for d in result['super'])} buys, {sum(d['sell_signal'] for d in result['super'])} sells")

        return result

    except Exception as e:
        return {
            'error': str(e),
            'status': 'error',
            'ticker': ticker,
            'message': f"Failed to process SuperTrend strategies for {ticker}"
        }

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # Display comparison plot
    # fig, df = supertrend_strategy_comparison("RELIANCE.NS")
    # Generate JSON-like strategy data
    data = get_supertrend_strategy_data("RELIANCE.NS", days=700, 
                                        high_vol_multiplier=3.0, 
                                        mid_vol_multiplier=2.0, 
                                        low_vol_multiplier=1.0,
                                        reversed_signals=True)
    # print(data)
