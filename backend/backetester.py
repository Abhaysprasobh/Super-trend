import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
import yfinance as yf
import datetime
from tqdm import tqdm
from copy import deepcopy

from supertrend import supertrend
from adaptive import adaptive_supertrend, historical_data

# Function to backtest trading strategies
def backtest_strategy(df, indicator_col, direction_col, initial_capital=100000, commission=0.001):
    """
    Backtest a trading strategy based on supertrend signals
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with price data and indicator signals
    indicator_col : str
        Column name for the indicator values
    direction_col : str
        Column name for direction values (1 for bullish, -1 for bearish)
    initial_capital : float
        Starting capital for the backtest
    commission : float
        Commission rate per trade (as a decimal)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with backtest results
    """
    backtest_df = df.copy()
    
    # Add required columns for backtest
    backtest_df['signal'] = 0  # 1 for buy, -1 for sell
    backtest_df['position'] = 0  # 1 for long, -1 for short, 0 for flat
    backtest_df['trade_price'] = np.nan
    backtest_df['capital'] = initial_capital
    backtest_df['holdings'] = 0
    backtest_df['cash'] = initial_capital
    backtest_df['portfolio_value'] = initial_capital
    backtest_df['returns'] = 0
    backtest_df['trade_id'] = 0
    
    # Generate signals based on direction changes
    backtest_df['prev_direction'] = backtest_df[direction_col].shift(1)
    
    # Signal when direction changes (we'll use Next Bar Open for entry/exit)
    backtest_df.loc[(backtest_df[direction_col] == 1) & 
                    (backtest_df['prev_direction'] == -1), 'signal'] = 1  # Buy signal
                    
    backtest_df.loc[(backtest_df[direction_col] == -1) & 
                    (backtest_df['prev_direction'] == 1), 'signal'] = -1  # Sell signal
    
    # Process signals and calculate positions
    position = 0
    trade_count = 0
    trades = []
    open_price = None
    open_date = None
    
    for i in range(1, len(backtest_df)):
        # Get today's data
        today = backtest_df.iloc[i]
        yesterday = backtest_df.iloc[i-1]
        today_signal = today['signal']
        
        # Carry forward position
        position = yesterday['position']
        
        # Update position based on signal
        if today_signal == 1:  # Buy signal
            if position <= 0:  # If we're flat or short
                # Close any existing position and go long
                position = 1
                open_price = today['Open']
                open_date = backtest_df.index[i]
                trade_count += 1
                
        elif today_signal == -1:  # Sell signal
            if position >= 0:  # If we're flat or long
                # Close any existing position and go flat
                if position == 1:  # If we were long, record the trade
                    trades.append({
                        'trade_id': trade_count,
                        'entry_date': open_date,
                        'entry_price': open_price,
                        'exit_date': backtest_df.index[i],
                        'exit_price': today['Open'],
                        'profit_pct': (today['Open'] / open_price - 1) * 100 - commission * 200,
                        'type': 'LONG'
                    })
                position = 0  # Go flat
                
        # Record position
        backtest_df.iloc[i, backtest_df.columns.get_loc('position')] = position
        
        # Calculate portfolio value
        if position == 1:  # Long position
            price = today['Close']
            prev_price = yesterday['Close']
            shares = yesterday['portfolio_value'] / prev_price
            holdings = shares * price
            returns = price / prev_price - 1
            portfolio_value = holdings
        else:  # Flat position
            returns = 0
            portfolio_value = yesterday['portfolio_value']
        
        # Update portfolio metrics
        backtest_df.iloc[i, backtest_df.columns.get_loc('holdings')] = holdings if position == 1 else 0
        backtest_df.iloc[i, backtest_df.columns.get_loc('portfolio_value')] = portfolio_value
        backtest_df.iloc[i, backtest_df.columns.get_loc('returns')] = returns
        backtest_df.iloc[i, backtest_df.columns.get_loc('trade_id')] = trade_count
    
    # Calculate cumulative returns
    backtest_df['cum_returns'] = (1 + backtest_df['returns']).cumprod() - 1
    
    # Convert trades to DataFrame
    trades_df = pd.DataFrame(trades)
    
    return backtest_df, trades_df


def calculate_performance_metrics(backtest_df, trades_df, benchmark_returns=None, risk_free_rate=0.03):
    """
    Calculate performance metrics from backtest results
    
    Parameters:
    -----------
    backtest_df : pandas.DataFrame
        DataFrame with backtest results
    trades_df : pandas.DataFrame
        DataFrame with trade details
    benchmark_returns : pandas.Series, optional
        Returns of a benchmark for comparison
    risk_free_rate : float
        Annualized risk-free rate for Sharpe ratio calculation
        
    Returns:
    --------
    dict
        Dictionary of performance metrics
    """
    # Drop rows with NaN values in returns
    returns = backtest_df['returns'].dropna()
    
    # Basic metrics
    total_return = (backtest_df['portfolio_value'].iloc[-1] / backtest_df['portfolio_value'].iloc[0]) - 1
    
    # Calculate trading days per year based on data frequency
    days_between = (backtest_df.index[-1] - backtest_df.index[0]).days
    trading_days = len(backtest_df)
    trading_days_per_year = trading_days / (days_between / 365)
    
    # Annualized return
    years = days_between / 365
    annualized_return = (1 + total_return) ** (1 / years) - 1
    
    # Risk metrics
    daily_returns = backtest_df['returns'].fillna(0)
    volatility = daily_returns.std() * (trading_days_per_year ** 0.5)
    sharpe = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
    
    # Drawdown calculation
    portfolio_values = backtest_df['portfolio_value']
    running_max = portfolio_values.cummax()
    drawdown = (portfolio_values / running_max - 1)
    max_drawdown = drawdown.min()
    
    # Win rate
    if len(trades_df) > 0:
        winning_trades = trades_df[trades_df['profit_pct'] > 0]
        win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0
        avg_win = winning_trades['profit_pct'].mean() if len(winning_trades) > 0 else 0
        avg_loss = trades_df[trades_df['profit_pct'] <= 0]['profit_pct'].mean() if len(trades_df[trades_df['profit_pct'] <= 0]) > 0 else 0
        profit_factor = abs(winning_trades['profit_pct'].sum() / trades_df[trades_df['profit_pct'] <= 0]['profit_pct'].sum()) if trades_df[trades_df['profit_pct'] <= 0]['profit_pct'].sum() != 0 else float('inf')
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
    
    # Calculate metrics vs benchmark if provided
    benchmark_metrics = {}
    if benchmark_returns is not None:
        # Ensure benchmark returns align with strategy returns
        aligned_returns = benchmark_returns.reindex(daily_returns.index).fillna(0)
        
        # Beta calculation
        covariance = daily_returns.cov(aligned_returns)
        benchmark_variance = aligned_returns.var()
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        # Alpha calculation (annualized)
        benchmark_total_return = (1 + aligned_returns).prod() - 1
        benchmark_annualized = (1 + benchmark_total_return) ** (1 / years) - 1
        alpha = annualized_return - (risk_free_rate + beta * (benchmark_annualized - risk_free_rate))
        
        benchmark_metrics = {
            'beta': beta,
            'alpha': alpha,
            'benchmark_return': benchmark_total_return,
            'benchmark_annualized': benchmark_annualized
        }
    
    # Combine all metrics
    metrics = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'total_trades': len(trades_df),
        'trading_days': trading_days,
        'years': years
    }
    
    # Add benchmark metrics if available
    if benchmark_metrics:
        metrics.update(benchmark_metrics)
    
    return metrics


def compare_strategies(results_dict, benchmark_returns=None):
    """
    Compare multiple backtested strategies
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary of strategy names and their backtest results
    benchmark_returns : pandas.Series, optional
        Returns of a benchmark for comparison
        
    Returns:
    --------
    dict
        Dictionary of comparison results
    """
    comparison = {}
    
    for strategy_name, (backtest_df, trades_df) in results_dict.items():
        metrics = calculate_performance_metrics(backtest_df, trades_df, benchmark_returns)
        comparison[strategy_name] = metrics
    
    # Convert to DataFrame for easier comparison
    comparison_df = pd.DataFrame(comparison)
    
    # Add a details section showing relative performance
    best_strategy = comparison_df.loc['total_return'].idxmax()
    
    return comparison_df, best_strategy


def plot_backtest_comparison(results_dict, metrics_df, title="Strategy Comparison"):
    """
    Plot the comparison of backtest results
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary of strategy names and their backtest results
    metrics_df : pandas.DataFrame
        DataFrame with performance metrics for each strategy
    title : str
        Title for the plot
    """
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, height_ratios=[3, 1, 1])
    
    # 1. Performance chart
    ax1 = plt.subplot(gs[0, :])
    
    # Plot each strategy's equity curve
    for strategy_name, (backtest_df, _) in results_dict.items():
        norm_equity = backtest_df['portfolio_value'] / backtest_df['portfolio_value'].iloc[0]
        ax1.plot(backtest_df.index, norm_equity, label=f"{strategy_name}")
    
    ax1.set_title("Strategy Performance Comparison", fontsize=16)
    ax1.set_ylabel("Growth of $1")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # 2. Drawdown chart
    ax2 = plt.subplot(gs[1, :], sharex=ax1)
    
    for strategy_name, (backtest_df, _) in results_dict.items():
        equity = backtest_df['portfolio_value']
        running_max = equity.cummax()
        drawdown = (equity / running_max - 1) * 100  # Convert to percentage
        ax2.fill_between(backtest_df.index, 0, drawdown, label=f"{strategy_name} DD", alpha=0.3)
    
    ax2.set_ylabel("Drawdown %")
    ax2.set_ylim(metrics_df.loc['max_drawdown'].min() * 110, 5)  # Add some padding
    ax2.grid(True, alpha=0.3)
    
    # 3. Position chart
    ax3 = plt.subplot(gs[2, 0], sharex=ax1)
    
    # Plot positions for each strategy
    for i, (strategy_name, (backtest_df, _)) in enumerate(results_dict.items()):
        # Create a stepped line for positions
        positions = backtest_df['position'].replace(0, np.nan)  # Replace flat positions with NaN
        ax3.step(backtest_df.index, i + positions * 0.4, where='post', 
                label=f"{strategy_name}", linewidth=2)
    
    ax3.set_title("Strategy Positions", fontsize=12)
    ax3.set_ylabel("Strategy")
    ax3.grid(True, alpha=0.3)
    ax3.set_yticks(range(len(results_dict)))
    ax3.set_yticklabels(results_dict.keys())
    
    # 4. Key metrics table
    ax4 = plt.subplot(gs[2, 1])
    ax4.axis('off')
    
    # Select key metrics to display
    key_metrics = ['total_return', 'annualized_return', 'sharpe_ratio', 
                  'max_drawdown', 'win_rate', 'total_trades']
    
    # Format metrics
    formatted_metrics = metrics_df.loc[key_metrics].copy()
    formatted_metrics.loc['total_return'] = formatted_metrics.loc['total_return'].apply(lambda x: f"{x*100:.2f}%")
    formatted_metrics.loc['annualized_return'] = formatted_metrics.loc['annualized_return'].apply(lambda x: f"{x*100:.2f}%")
    formatted_metrics.loc['sharpe_ratio'] = formatted_metrics.loc['sharpe_ratio'].apply(lambda x: f"{x:.2f}")
    formatted_metrics.loc['max_drawdown'] = formatted_metrics.loc['max_drawdown'].apply(lambda x: f"{x*100:.2f}%")
    formatted_metrics.loc['win_rate'] = formatted_metrics.loc['win_rate'].apply(lambda x: f"{x*100:.1f}%")
    formatted_metrics.loc['total_trades'] = formatted_metrics.loc['total_trades'].apply(lambda x: f"{int(x)}")
    
    # Create a nice looking table
    table = ax4.table(
        cellText=formatted_metrics.values,
        rowLabels=['Total Return', 'Ann. Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate', 'Total Trades'],
        colLabels=formatted_metrics.columns,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Highlight the best strategy in each metric
    for i, metric in enumerate(key_metrics):
        if metric in ['max_drawdown']:  # Lower is better
            best_col = metrics_df.loc[metric].idxmin()
        else: 
            best_col = metrics_df.loc[metric].idxmax()
        
        col_idx = list(formatted_metrics.columns).index(best_col)
        cell = table.get_celld()[i+1, col_idx]
        cell.set_facecolor('#c9f7c9')  # Light green
    
    # Improve x-axis date formatting
    plt.gcf().autofmt_xdate()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    plt.tight_layout()
    return fig


def backtest_supertrend_strategies(df, strategies_config, initial_capital=100000):
    """
    Backtest multiple SuperTrend strategies on the same data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with OHLC data
    strategies_config : dict
        Dictionary with strategy configurations
    initial_capital : float
        Initial capital for backtesting
        
    Returns:
    --------
    dict
        Dictionary of backtest results for each strategy
    """
    results = {}
    
    # Make a copy of the DataFrame to avoid modifying the original
    data = df.copy()
    
    for strategy_name, config in tqdm(strategies_config.items(), desc="Backtesting strategies"):
        # Apply strategy-specific indicator
        if config.get('type') == 'standard':
            # Regular SuperTrend
            result_df = supertrend(
                data, 
                length=config.get('length', 10), 
                multiplier=config.get('multiplier', 3.0)
            )
            indicator_col = f"SUPERT_{config.get('length', 10)}_{config.get('multiplier', 3.0)}"
            direction_col = f"SUPERTd_{config.get('length', 10)}_{config.get('multiplier', 3.0)}"
            
        elif config.get('type') == 'adaptive':
            # Adaptive SuperTrend
            result_df = adaptive_supertrend(
                data,
                atr_len=config.get('atr_len', 10),
                factor=config.get('factor', 3.0),
                training_data_period=config.get('training_period', 100)
            )
            indicator_col = 'ADAPT_SUPERT'
            direction_col = 'ADAPT_SUPERTd'
        
        else:
            print(f"Unknown strategy type: {config.get('type')}")
            continue
        
        # Run backtest
        backtest_result, trades = backtest_strategy(
            result_df, 
            indicator_col=indicator_col,
            direction_col=direction_col,
            initial_capital=initial_capital
        )
        
        # Store results
        results[strategy_name] = (backtest_result, trades)
    
    return results


def plot_supertrend_signals(df, strategy_type='standard', title=None, figsize=(15, 8)):
    """
    Plot SuperTrend signals on price chart
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with price data and SuperTrend indicator values
    strategy_type : str
        Type of SuperTrend strategy ('standard' or 'adaptive')
    title : str, optional
        Title for the plot
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    fig = plt.figure(figsize=figsize)
    
    if strategy_type == 'standard':
        # Find SuperTrend columns
        st_cols = [col for col in df.columns if col.startswith('SUPERT_')]
        if not st_cols:
            raise ValueError("No SuperTrend columns found in the DataFrame")
            
        indicator_col = st_cols[0]
        direction_col = f"SUPERTd{indicator_col[6:]}"  # Convert SUPERT_10_3.0 to SUPERTd_10_3.0
        
    elif strategy_type == 'adaptive':
        indicator_col = 'supertrend'
        direction_col = 'direction'
        
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    # Create plot
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1], sharex=ax1)
    
    # Plot price
    ax1.plot(df.index, df['Close'], label='Close', color='black', alpha=0.7)
    
    # Plot SuperTrend
    bullish = df[direction_col] == 1
    bearish = df[direction_col] == -1
    
    ax1.plot(df.index[bullish], df[indicator_col][bullish], 
             color='green', linewidth=1.5, label='SuperTrend (Bullish)')
    ax1.plot(df.index[bearish], df[indicator_col][bearish], 
             color='red', linewidth=1.5, label='SuperTrend (Bearish)')
    
    # Plot signals
    signals = df['signal'] != 0
    buy_signals = (df['signal'] == 1) & signals
    sell_signals = (df['signal'] == -1) & signals
    
    ax1.scatter(df.index[buy_signals], df['Low'][buy_signals] * 0.99, 
                color='green', marker='^', s=100, label='Buy Signal')
    ax1.scatter(df.index[sell_signals], df['High'][sell_signals] * 1.01, 
                color='red', marker='v', s=100, label='Sell Signal')
    
    # Plot position
    ax2.step(df.index, df['position'], where='post', color='blue', label='Position')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.fill_between(df.index, df['position'], 0, where=df['position'] > 0, 
                     color='green', alpha=0.3, step='post')
    ax2.fill_between(df.index, df['position'], 0, where=df['position'] < 0, 
                     color='red', alpha=0.3, step='post')
    
    # Set titles and labels
    if title:
        ax1.set_title(title, fontsize=16)
    else:
        ax1.set_title(f"SuperTrend Strategy ({strategy_type.capitalize()})", fontsize=16)
        
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    ax2.set_ylabel('Position')
    ax2.set_yticks([-1, 0, 1])
    ax2.set_yticklabels(['Short', 'Flat', 'Long'])
    ax2.grid(True, alpha=0.3)
    
    # Improve x-axis date formatting
    plt.gcf().autofmt_xdate()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    plt.tight_layout()
    return fig


def plot_adaptive_supertrend_detail(df, title="Adaptive SuperTrend Analysis"):
    """
    Plot detailed analysis of Adaptive SuperTrend including volatility clusters
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with Adaptive SuperTrend results
    title : str
        Title for the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.1)
    
    # 1. Price and SuperTrend
    ax1 = plt.subplot(gs[0])
    
    # Plot price
    ax1.plot(df.index, df['Close'], label='Close', color='black', alpha=0.7)
    
    # Plot SuperTrend
    bullish = df['direction'] == 1
    bearish = df['direction'] == -1
    
    ax1.plot(df.index[bullish], df['supertrend'][bullish], 
             color='green', linewidth=1.5, label='SuperTrend (Bullish)')
    ax1.plot(df.index[bearish], df['supertrend'][bearish], 
             color='red', linewidth=1.5, label='SuperTrend (Bearish)')
    
    # Plot signals
    signals = df['signal'] != 0
    buy_signals = (df['signal'] == 1) & signals
    sell_signals = (df['signal'] == -1) & signals
    
    ax1.scatter(df.index[buy_signals], df['Low'][buy_signals] * 0.99, 
                color='green', marker='^', s=100, label='Buy Signal')
    ax1.scatter(df.index[sell_signals], df['High'][sell_signals] * 1.01, 
                color='red', marker='v', s=100, label='Sell Signal')
    
    # 2. Volatility and Centroids
    ax2 = plt.subplot(gs[1], sharex=ax1)
    
    ax2.plot(df.index, df['volatility'], label='ATR (Volatility)', color='purple', alpha=0.7)
    
    # Plot centroids
    if 'high_vol_centroid' in df.columns:
        ax2.plot(df.index, df['high_vol_centroid'], label='High Vol', color='red', linestyle='--', alpha=0.7)
        ax2.plot(df.index, df['mid_vol_centroid'], label='Mid Vol', color='orange', linestyle='--', alpha=0.7)
        ax2.plot(df.index, df['low_vol_centroid'], label='Low Vol', color='green', linestyle='--', alpha=0.7)
    
    # 3. Volatility Cluster/Regime
    ax3 = plt.subplot(gs[2], sharex=ax1)
    
    # Define colors and labels for volatility levels
    colors = {0: 'red', 1: 'orange', 2: 'green'}
    labels = {0: 'High', 1: 'Medium', 2: 'Low'}
    
    # Create colored rectangles for each cluster value
    if 'cluster' in df.columns:
        for i in range(len(df)-1):
            if pd.isna(df['cluster'].iloc[i]):
                continue
                
            cluster = int(df['cluster'].iloc[i])
            start_date = df.index[i]
            
            # Find next date or end of dataframe
            if i+1 < len(df):
                end_date = df.index[i+1]
            else:
                end_date = start_date + pd.Timedelta(days=1)  # Just add a day for visualization
                
            # Convert dates to numbers for plotting
            start_num = mdates.date2num(start_date)
            end_num = mdates.date2num(end_date)
            
            # Add rectangle patch
            rect = plt.Rectangle((start_num, cluster-0.4), end_num-start_num, 0.8, 
                                color=colors.get(cluster, 'gray'), alpha=0.7)
            ax3.add_patch(rect)
        
        # Create custom legend
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label='High Volatility'),
            Patch(facecolor='orange', alpha=0.7, label='Medium Volatility'),
            Patch(facecolor='green', alpha=0.7, label='Low Volatility')
        ]
        ax3.legend(handles=legend_elements, loc='upper right')
    
    # 4. Position
    ax4 = plt.subplot(gs[3], sharex=ax1)
    
    ax4.step(df.index, df['position'], where='post', color='blue', label='Position')
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.fill_between(df.index, df['position'], 0, where=df['position'] > 0, 
                     color='green', alpha=0.3, step='post')
    ax4.fill_between(df.index, df['position'], 0, where=df['position'] < 0, 
                     color='red', alpha=0.3, step='post')
    
    # Set titles and labels
    ax1.set_title(title, fontsize=16)
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    ax2.set_ylabel('Volatility')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=8)
    
    # Set y-ticks and labels for volatility regimes
    ax3.set_ylim(-0.5, 2.5)
    ax3.set_yticks([0, 1, 2])
    ax3.set_yticklabels(['High', 'Medium', 'Low'])
    ax3.set_ylabel('Volatility\nRegime')
    ax3.grid(True, alpha=0.3)
    
    ax4.set_ylabel('Position')
    ax4.set_yticks([-1, 0, 1])
    ax4.set_yticklabels(['Short', 'Flat', 'Long'])
    ax4.grid(True, alpha=0.3)
    
    # Improve x-axis date formatting
    plt.gcf().autofmt_xdate()
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    plt.tight_layout()
    return fig


def run_full_comparison(symbol, strategies, interval='1d', days=700, initial_capital=100000):
    """
    Run a full comparison between standard and adaptive SuperTrend strategies and return results as CSV files
    
    Parameters:
    -----------
    symbol : str
        Ticker symbol to analyze
    interval : str
        Data interval ('1d', '1h', etc.)
    days : int
        Number of days of historical data
    initial_capital : float
        Initial capital for backtesting
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'performance': Performance metrics DataFrame as CSV string
        - 'equity_curves': Dictionary of equity curves as CSV strings
        - 'trades': Dictionary of trade lists as CSV strings
        - 'best_strategy': Name of best performing strategy
    """
    print(f"Downloading historical data for {symbol}...")
    df = historical_data(symbol, interval, days)
    
    # Define strategy configurations
    # strategies = {
    #     "SuperTrend (10,3)": {
    #         "type": "standard",
    #         "length": 10,
    #         "multiplier": 3.0
    #     },
    #     "Adaptive SuperTrend": {
    #         "type": "adaptive",
    #         "atr_len": 10,
    #         "factor": 3.0,
    #         "training_period": 75
    #     }
    # }
    
    # Run backtests
    print("Running backtests...")
    results = backtest_supertrend_strategies(df, strategies, initial_capital)
    
    # Calculate benchmark returns
    benchmark_returns = df['Close'].pct_change()
    
    # Compare strategies
    comparison_df, best_strategy = compare_strategies(results, benchmark_returns)
    
    # Prepare return data
    return_data = {
        'performance': comparison_df.to_csv(),
        'equity_curves': {},
        'trades': {},
        'best_strategy': best_strategy
    }

    
    
    # Add equity curves and trades for each strategy
    for strategy_name, (backtest_df, trades_df) in results.items():
        # Normalize equity curve to start at 100
        norm_equity = backtest_df['portfolio_value'] / backtest_df['portfolio_value'].iloc[0] * 100
        equity_df = pd.DataFrame({
            'date': backtest_df.index,
            'equity': norm_equity,
            'drawdown': (backtest_df['portfolio_value'] / backtest_df['portfolio_value'].cummax() - 1) * 100
        })
        
        return_data['equity_curves'][strategy_name] = equity_df.to_csv(index=False)
        return_data['trades'][strategy_name] = trades_df.to_csv(index=False)
    
    return return_data


if __name__ == "__main__":
    symbol = "PLTR"  
    strategies = {
        "SuperTrend (10,3)": {
            "type": "standard",
            "length": 10,
            "multiplier": 3.0
        },
        "Adaptive SuperTrend": {
            "type": "adaptive",
            "atr_len": 10,
            "factor": 3.0,
            "training_period": 75
        }
    }
    results = run_full_comparison(symbol, strategies=strategies)
    
    print("\nPerformance metrics:")
    print(results['performance'])
    
    print(f"\nBest strategy: {results['best_strategy']}")
    
    print("\nSample of equity curve for best strategy:")
    print(results['equity_curves'][results['best_strategy']][:200])