

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

def supertrend(df, length=7, multiplier=3.0, high='High', low='Low', close='Close', append=True):
    df = df.copy()
    
    # Calculate True Range (TR) & ATR (RMA)
    tr1 = df[high] - df[low]
    tr2 = abs(df[high] - df[close].shift())
    tr3 = abs(df[low] - df[close].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/length, adjust=False).mean()  # RMA

    # HL2 and initial bands
    hl2 = (df[high] + df[low]) / 2
    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr

    direction = pd.Series(1, index=df.index)  # Start bullish
    trend = pd.Series(np.nan, index=df.index)

    for i in range(1, len(df)):
        prev_dir = direction.iloc[i-1]
        prev_upper = upperband.iloc[i-1]
        prev_lower = lowerband.iloc[i-1]
        curr_close = df[close].iloc[i]

        if curr_close > prev_upper:
            direction.iloc[i] = 1
        elif curr_close < prev_lower:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = prev_dir
            # Lock bands only if direction persists
            if direction.iloc[i] == 1 and lowerband.iloc[i] < prev_lower:
                lowerband.iloc[i] = prev_lower
            elif direction.iloc[i] == -1 and upperband.iloc[i] > prev_upper:
                upperband.iloc[i] = prev_upper

        trend.iloc[i] = lowerband.iloc[i] if direction.iloc[i] == 1 else upperband.iloc[i]

    # Invalidate first 'length' periods
    trend[:length] = np.nan

    # Append results to DataFrame
    suffix = f"_{length}_{multiplier}"
    df[f"SUPERT{suffix}"] = trend
    df[f"SUPERTd{suffix}"] = direction
    df[f"SUPERTl{suffix}"] = trend.where(direction == 1)
    df[f"SUPERTs{suffix}"] = trend.where(direction == -1)
    return df
def plot_standard_supertrend(stock_name, df, length=7, multiplier=3.0, fill_alpha=0.25):
    """
    Plot standard SuperTrend as a single continuous line, color-coded by trend direction.

    Parameters:
    -----------
    stock_name : str
        Stock name for the plot title
    df : pandas.DataFrame
        DataFrame with SuperTrend values
    length : int
        Period length used in the SuperTrend
    multiplier : float
        ATR multiplier used in the SuperTrend
    fill_alpha : float
        Alpha for the filled region between price and SuperTrend
    """


    suffix = f"_{length}_{multiplier}"
    trend_col = f"SUPERT{suffix}"
    dir_col = f"SUPERTd{suffix}"

    plot_df = df.dropna(subset=[trend_col, dir_col]).copy()
    trend = plot_df[trend_col].values
    direction = plot_df[dir_col].values

    dates = mdates.date2num(plot_df.index)
    close = plot_df['Close'].values

    # Prep for fill between trend and price
    polys_bullish = []
    polys_bearish = []

    for i in range(len(plot_df) - 1):
        x1, x2 = dates[i], dates[i+1]
        y1_price, y2_price = close[i], close[i+1]
        y1_trend, y2_trend = trend[i], trend[i+1]

        if direction[i] == 1:  # Bullish: trend under price, fill red
            y2_trend = trend[i+1] if direction[i+1] == 1 else y2_price
            poly = [(x1, y1_trend), (x1, y1_price), (x2, y2_price), (x2, y2_trend)]
            polys_bullish.append(plt.Polygon(poly))
        else:  # Bearish: trend above price, fill green
            y2_trend = trend[i+1] if direction[i+1] == -1 else y2_price
            poly = [(x1, y1_price), (x1, y1_trend), (x2, y2_trend), (x2, y2_price)]
            polys_bearish.append(plt.Polygon(poly))

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(plot_df.index, plot_df['Close'], label='Close', color='black', alpha=0.7)

    if polys_bullish:
        ax.add_collection(PatchCollection(polys_bullish, facecolor='red', alpha=fill_alpha))
    if polys_bearish:
        ax.add_collection(PatchCollection(polys_bearish, facecolor='green', alpha=fill_alpha))

    # Split SuperTrend line by direction
    for i in range(1, len(trend)):
        color = 'green' if direction[i] == 1 else 'red'
        ax.plot(dates[i-1:i+1], trend[i-1:i+1], color=color, linewidth=1.5)

    # Mark trend changes
    bull_cross = (plot_df[dir_col].shift() == -1) & (plot_df[dir_col] == 1)
    bear_cross = (plot_df[dir_col].shift() == 1) & (plot_df[dir_col] == -1)

    ax.scatter(plot_df.index[bull_cross], plot_df[trend_col][bull_cross] * 0.99,
               color='green', marker='^', s=120, label='Buy Signal')
    ax.scatter(plot_df.index[bear_cross], plot_df[trend_col][bear_cross] * 1.01,
               color='red', marker='v', s=120, label='Sell Signal')

    # Optional cluster labels
    if 'cluster' in plot_df.columns:
        cluster_colors = {0: 'orange', 1: 'purple', 2: 'blue'}
        clusters = plot_df['cluster'].dropna().astype(int).unique()
        for cluster in clusters:
            chg = (plot_df['cluster'] == cluster) & (plot_df['cluster'].shift() != cluster)
            for idx in np.where(chg)[0]:
                date = plot_df.index[idx]
                val = plot_df[trend_col].iloc[idx]
                ax.text(date, val, f"{cluster+1}", ha='center', va='center',
                        color=cluster_colors.get(cluster, 'gray'),
                        bbox=dict(facecolor='white', edgecolor=cluster_colors.get(cluster, 'gray'), alpha=0.7))

    ax.set_title(f'SuperTrend Indicator ({stock_name})', fontsize=16)
    ax.set_ylabel('Price')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_ylim(plot_df['Low'].min() * 0.98, plot_df['High'].max() * 1.02)

    plt.tight_layout()
    return fig




# Example usage on historical data:
if __name__ == '__main__':
    from analysis import historical_data
    stock_name = 'RELIANCE.NS'
    data = historical_data(stock_name, '1d', 700)

    result = supertrend(data, length=7, multiplier=3.0, append=True)
    result.to_excel(f'{stock_name}_supertrend.xlsx')
    fig = plot_standard_supertrend(stock_name, result)
    plt.show()

    print(result)
