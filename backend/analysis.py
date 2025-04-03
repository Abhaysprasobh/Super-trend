

import pandas as pd
import numpy as np
import yfinance as yf
import datetime
from copy import deepcopy

from supertrend import supertrend
from adaptive import adaptive_supertrend, historical_data

from backetester import backtest_supertrend_strategies, compare_strategies
def get_indicator_data(symbol, interval='1d', days=700):
    """
    Fetch historical data and calculate both standard and adaptive SuperTrend indicators
    
    Parameters:
    -----------
    symbol : str
        Ticker symbol to analyze
    interval : str
        Data interval ('1d', '1h', etc.)
    days : int
        Number of days of historical data
        
    Returns:
    --------
    dict
        Dictionary containing indicator data and price history
    """
    # Get historical data
    df = historical_data(symbol, interval, days)
    
    # Calculate standard SuperTrend (10,3)
    st_df = supertrend(df.copy(), length=10, multiplier=3.0)
    st_indicator_col = f"SUPERT_10_3.0"
    st_direction_col = f"SUPERTd_10_3.0"
    
    # Calculate Adaptive SuperTrend
    adaptive_df = adaptive_supertrend(
        df.copy(),
        atr_len=10,
        factor=3.0,
        training_data_period=75
    )
    
    # Merge the indicators together
    result_df = df.copy()
    result_df[st_indicator_col] = st_df[st_indicator_col]
    result_df[st_direction_col] = st_df[st_direction_col]
    result_df['adaptive_supertrend'] = adaptive_df['supertrend']
    result_df['adaptive_direction'] = adaptive_df['direction']
    
    # Convert to dictionary of lists for JSON serialization
    result_dict = {
        'dates': result_df.index.strftime('%Y-%m-%d').tolist(),
        'open': result_df['Open'].tolist(),
        'high': result_df['High'].tolist(),
        'low': result_df['Low'].tolist(),
        'close': result_df['Close'].tolist(),
        'volume': result_df['Volume'].tolist(),
        'supertrend': result_df[st_indicator_col].tolist(),
        'supertrend_direction': result_df[st_direction_col].tolist(),
        'adaptive_supertrend': result_df['adaptive_supertrend'].tolist(),
        'adaptive_direction': result_df['adaptive_direction'].tolist()
    }
    
    return result_dict

# Function to backtest strategies and get performance metrics
def backtest_comparison(symbol, interval='1d', days=700, initial_capital=100000):
    """
    Run backtest for both standard and adaptive SuperTrend strategies
    
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
        Dictionary containing performance results and trade signals
    """
    # Get historical data
    df = historical_data(symbol, interval, days)
    
    # Define strategies
    strategies = {
        "SuperTrend": {
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
    
    # Run backtests
    results = backtest_supertrend_strategies(df, strategies, initial_capital)
    
    # Calculate benchmark returns
    benchmark_returns = df['Close'].pct_change()
    
    # Compare strategies
    comparison_df, best_strategy = compare_strategies(results, benchmark_returns)
    
    # Format performance metrics for frontend
    performance_metrics = format_performance_metrics(comparison_df)
    
    # Extract buy/sell signals for frontend
    signals = extract_trade_signals(results)
    
    # Create equity curves data for charts
    equity_curves = create_equity_curves_data(results)
    
    # Annual returns breakdown
    annual_returns = calculate_annual_returns(results)
    
    return {
        'metrics': performance_metrics,
        'signals': signals,
        'equity_curves': equity_curves,
        'annual_returns': annual_returns,
        'best_strategy': best_strategy
    }

def format_performance_metrics(comparison_df):
    """Format performance metrics for frontend display"""
    # Select key metrics to display
    key_metrics = [
        'total_return', 'annualized_return', 'sharpe_ratio', 
        'max_drawdown', 'win_rate', 'total_trades'
    ]
    
    metrics = comparison_df.loc[key_metrics].copy()
    
    # Convert to proper format for frontend
    formatted = {}
    for strategy in metrics.columns:
        formatted[strategy] = {
            'total_return': f"{metrics.loc['total_return', strategy]*100:.2f}%",
            'annualized_return': f"{metrics.loc['annualized_return', strategy]*100:.2f}%",
            'sharpe_ratio': f"{metrics.loc['sharpe_ratio', strategy]:.2f}",
            'max_drawdown': f"{metrics.loc['max_drawdown', strategy]*100:.2f}%",
            'win_rate': f"{metrics.loc['win_rate', strategy]*100:.1f}%",
            'total_trades': int(metrics.loc['total_trades', strategy])
        }
        
        # Also include raw values for sorting/comparisons
        formatted[strategy]['total_return_value'] = float(metrics.loc['total_return', strategy])
        formatted[strategy]['annualized_return_value'] = float(metrics.loc['annualized_return', strategy])
    
    return formatted

def extract_trade_signals(results):
    """Extract buy/sell signals from backtest results"""
    signals = {}
    
    for strategy_name, (backtest_df, _) in results.items():
        # Find buy and sell signals
        signal_df = backtest_df[backtest_df['signal'] != 0].copy()
        
        signals[strategy_name] = {
            'dates': signal_df.index.strftime('%Y-%m-%d').tolist(),
            'signals': signal_df['signal'].tolist(),
            'prices': signal_df['Open'].tolist()
        }
    
    return signals

def create_equity_curves_data(results):
    """Create equity curves data for charting"""
    equity_data = {}
    
    for strategy_name, (backtest_df, _) in results.items():
        # Normalize equity curve to start at 100
        equity_data[strategy_name] = {
            'dates': backtest_df.index.strftime('%Y-%m-%d').tolist(),
            'equity': (backtest_df['portfolio_value'] / backtest_df['portfolio_value'].iloc[0] * 100).tolist(),
            'drawdown': (backtest_df['portfolio_value'] / backtest_df['portfolio_value'].cummax() - 1 * 100).tolist()
        }
    
    return equity_data

def calculate_annual_returns(results):
    """Calculate annual returns for each strategy"""
    annual_returns = {}
    
    for strategy_name, (backtest_df, _) in results.items():
        # Resample returns to yearly frequency
        yearly_returns = backtest_df['portfolio_value'].resample('Y').last().pct_change()
        yearly_returns.iloc[0] = backtest_df['portfolio_value'].iloc[-1] / backtest_df['portfolio_value'].iloc[0] - 1
        
        annual_returns[strategy_name] = {
            'years': [year.strftime('%Y') for year in yearly_returns.index],
            'returns': [round(ret * 100, 2) for ret in yearly_returns.values]
        }
    
    return annual_returns

# Function to backtest trading strategies - unchanged from original
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


def api_get_indicator_comparison(symbol, interval='1d', days=700):
    """
    API endpoint function to get indicator comparison data
    
    Parameters:
    -----------
    symbol : str
        Ticker symbol to analyze
    interval : str
        Data interval ('1d', '1h', etc.)
    days : int
        Number of days of historical data
        
    Returns:
    --------
    dict
        Dictionary with all data needed for frontend visualization
    """
    # Get indicator data
    indicator_data = get_indicator_data(symbol, interval, days)
    
    # Get backtest results
    backtest_results = backtest_comparison(symbol, interval, days)
    
    # Combine results
    result = {
        'symbol': symbol,
        # 'indicator_data': indicator_data,
        'performance': backtest_results['metrics'],
        'signals': backtest_results['signals'],
        # 'equity_curves': backtest_results['equity_curves'],
        'annual_returns': backtest_results['annual_returns'],
        'best_strategy': backtest_results['best_strategy']
    }
    
    return result
def calculate_annual_returns(results):
    """Calculate annual returns for each strategy"""
    annual_returns = {}
    
    for strategy_name, (backtest_df, _) in results.items():
        # Get yearly returns
        yearly_returns = pd.DataFrame()
        yearly_returns['returns'] = backtest_df['portfolio_value'].resample('Y').last().pct_change()
        # Calculate first year return properly
        first_year_return = (backtest_df['portfolio_value'].iloc[-1] / 
                           backtest_df['portfolio_value'].iloc[0] - 1)
        yearly_returns.iloc[0] = first_year_return
        
        annual_returns[strategy_name] = {
            'years': yearly_returns.index.strftime('%Y').tolist(),
            'returns': [float(x) * 100 for x in yearly_returns['returns'].values]
        }
    
    return annual_returns

def format_performance_metrics(comparison_df):
    """Format performance metrics for frontend display"""
    key_metrics = [
        'total_return', 'annualized_return', 'sharpe_ratio', 
        'max_drawdown', 'win_rate', 'total_trades'
    ]
    
    formatted = {}
    for strategy in comparison_df.columns:
        metrics = comparison_df[strategy]
        formatted[strategy] = {
            'total_return': f"{metrics['total_return']*100:.2f}%",
            'annualized_return': f"{metrics['annualized_return']*100:.2f}%",
            'sharpe_ratio': f"{metrics['sharpe_ratio']:.2f}",
            'max_drawdown': f"{metrics['max_drawdown']*100:.2f}%",
            'win_rate': f"{metrics['win_rate']*100:.1f}%",
            'total_trades': int(metrics['total_trades']),
            # Raw values for sorting
            'total_return_value': float(metrics['total_return']),
            'annualized_return_value': float(metrics['annualized_return'])
        }
    
    return formatted

def extract_trade_signals(results):
    """Extract buy/sell signals from backtest results"""
    signals = {}
    
    for strategy_name, (backtest_df, _) in results.items():
        signal_df = backtest_df[backtest_df['signal'] != 0].copy()
        signals[strategy_name] = {
            'dates': signal_df.index.strftime('%Y-%m-%d').tolist(),
            'signals': signal_df['signal'].tolist(),
            'prices': signal_df['Open'].tolist()
        }
    
    return signals

def create_equity_curves_data(results):
    """Create equity curves data for charting"""
    equity_data = {}
    
    for strategy_name, (backtest_df, _) in results.items():
        norm_equity = backtest_df['portfolio_value'] / backtest_df['portfolio_value'].iloc[0] * 100
        drawdown = (backtest_df['portfolio_value'] / 
                   backtest_df['portfolio_value'].cummax() - 1) * 100
        
        equity_data[strategy_name] = {
            'dates': backtest_df.index.strftime('%Y-%m-%d').tolist(),
            'equity': norm_equity.tolist(),
            'drawdown': drawdown.tolist()
        }
    
    return equity_data


def main():
    # Example parameters
    symbol = "AAPL"  # Example stock symbol
    interval = "1d"  # Daily interval
    days = 700      # Number of days of historical data

    # Get indicator comparison data
    print(f"\nAnalyzing {symbol} for the last {days} days...\n")
    
    try:
        # Get complete analysis using api_get_indicator_comparison
        results = api_get_indicator_comparison(symbol, interval, days)
        
        # Print basic information
        print(f"Symbol: {results['symbol']}")
        
        # Print performance metrics for each strategy
        print("\nPerformance Metrics:")
        print("-" * 50)
        for strategy, metrics in results['performance'].items():
            print(f"\n{strategy}:")
            print(f"Total Return: {metrics['total_return']}")
            print(f"Annualized Return: {metrics['annualized_return']}")
            print(f"Sharpe Ratio: {metrics['sharpe_ratio']}")
            print(f"Max Drawdown: {metrics['max_drawdown']}")
            print(f"Win Rate: {metrics['win_rate']}")
            print(f"Total Trades: {metrics['total_trades']}")
        
        # Print annual returns
        print("\nAnnual Returns:")
        print("-" * 50)
        for strategy, data in results['annual_returns'].items():
            print(f"\n{strategy}:")
            for year, ret in zip(data['years'], data['returns']):
                print(f"{year}: {ret:.2f}%")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
