

import pandas as pd
import numpy as np

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


# Example usage on historical data:
if __name__ == '__main__':
    from test import historical_data
    data = historical_data('AAPL', '1d', 700)

    result = supertrend(data, length=7, multiplier=3.0, append=True)
    print(result)
