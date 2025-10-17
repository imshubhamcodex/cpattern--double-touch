import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta
import pytz

# === Fetch Data ===
def fetch_nifty_data(interval=5, days_back=30):
    ist = pytz.timezone('Asia/Kolkata')
    now_ist = datetime.now(ist)
    start_time_ist = now_ist - timedelta(days=days_back)
    end_time_ms = int(now_ist.timestamp() * 1000)
    start_time_ms = int(start_time_ist.timestamp() * 1000)

    url = (
        f"https://groww.in/v1/api/charting_service/v4/chart/exchange/NSE/segment/CASH/NIFTY"
        f"?endTimeInMillis={end_time_ms}&intervalInMinutes={interval}&startTimeInMillis={start_time_ms}"
    )
    r = requests.get(url)
    data = r.json()["candles"]

    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df.drop(columns=["volume"], inplace=True)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s").dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
    return df.reset_index(drop=True)


# === Swing Detection (matches Pine Script logic) ===
def detect_swing(df, start_idx, end_idx, swing_threshold):
    length = end_idx - start_idx + 1
    if length <= 0:
        return False

    swing_high = df.loc[start_idx, "low"]
    swing_low = df.loc[start_idx, "high"]

    for i in range(start_idx, end_idx + 1):
        swing_high = max(swing_high, df.loc[i, "high"])
        swing_low = min(swing_low, df.loc[i, "low"])

    return (swing_high - swing_low) > swing_threshold


# === Main Candle Detection (forward iteration) ===
def find_marked_candles(df, u_wick_threshold, l_wick_threshold, buffer_window, wick_diff, st, stt):
    # Calculate wick sizes
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

    # Containers
    mark_up, test_up, trigger_up, mark_up_dup = [], [], [], []
    mark_dn, test_dn, trigger_dn, mark_dn_dup = [], [], [], []

    # Iterate forward through candles
    for i in range(len(df)):
        upper_wick = df.loc[i, "upper_wick"]
        lower_wick = df.loc[i, "lower_wick"]

        # --- Detect Marked Candles (UP) ---
        if upper_wick >= u_wick_threshold:
            for j in range(i + 1, len(df)):
                diff = df.loc[j, "high"] - df.loc[i, "high"]
                if diff > 0:
                    break
                elif abs(diff) <= wick_diff and j >= i + buffer_window:
                    upper_wick_j = df.loc[j, "high"] - max(df.loc[j, "open"], df.loc[j, "close"])
                    if upper_wick_j >= u_wick_threshold:
                        if detect_swing(df, i, j, st):
                            mark_up.append(j)
                            test_up.append(i)

        # --- Detect Marked Candles (DOWN) ---
        if lower_wick >= l_wick_threshold:
            for j in range(i + 1, len(df)):
                diff = df.loc[i, "low"] - df.loc[j, "low"]
                if diff > 0:
                    break
                elif abs(diff) <= wick_diff and j >= i + buffer_window:
                    lower_wick_j = min(df.loc[j, "open"], df.loc[j, "close"]) - df.loc[j, "low"]
                    if lower_wick_j >= l_wick_threshold:
                        if detect_swing(df, i, j, st):
                            mark_dn.append(j)
                            test_dn.append(i)

    # --- Find Trigger Candles (UP) ---
    for idx in mark_up:
        for j in range(idx + 1, len(df)):
            diff = df.loc[j, "high"] - df.loc[idx, "high"]
            if diff > 0 and j < idx + buffer_window:
                break
            elif diff > 0 and j >= idx + buffer_window:
                if detect_swing(df, idx, j, stt):
                    trigger_up.append(j)
                    mark_up_dup.append(idx)
                break

    # --- Find Trigger Candles (DOWN) ---
    for idx in mark_dn:
        for j in range(idx + 1, len(df)):
            diff = df.loc[idx, "low"] - df.loc[j, "low"]
            if diff > 0 and j < idx + buffer_window:
                break
            elif diff > 0 and j >= idx + buffer_window:
                if detect_swing(df, idx, j, stt):
                    trigger_dn.append(j)
                    mark_dn_dup.append(idx)
                break

    return {
        "mark_candles_up": mark_up_dup,
        "test_candles_origin_up": test_up,
        "trigger_candles_up": trigger_up,
        "mark_candles_down": mark_dn_dup,
        "test_candles_origin_down": test_dn,
        "trigger_candles_down": trigger_dn
    }


# === Visualization ===
def plot_candlestick_with_marks(df, results):
    plt.figure(figsize=(15, 8))
    for i in range(len(df)):
        color = 'green' if df.loc[i, "close"] >= df.loc[i, "open"] else 'red'
        plt.plot([i, i], [df.loc[i, "low"], df.loc[i, "high"]], color='black', linewidth=1)
        plt.plot([i, i], [df.loc[i, "open"], df.loc[i, "close"]], color=color, linewidth=3)

    # Plot markers
    if results["mark_candles_up"]:
        plt.scatter(results["mark_candles_up"], df.loc[results["mark_candles_up"], "low"] - 5,
                    marker="^", color="orange", s=80, label="Mark Up")
    if results["trigger_candles_up"]:
        plt.scatter(results["trigger_candles_up"], df.loc[results["trigger_candles_up"], "low"] - 8,
                    marker="^", color="green", s=80, label="Trigger Up")
    if results["mark_candles_down"]:
        plt.scatter(results["mark_candles_down"], df.loc[results["mark_candles_down"], "high"] + 5,
                    marker="v", color="orange", s=80, label="Mark Down")
    if results["trigger_candles_down"]:
        plt.scatter(results["trigger_candles_down"], df.loc[results["trigger_candles_down"], "high"] + 8,
                    marker="v", color="red", s=80, label="Trigger Down")

    plt.title("NIFTY Marked Candles (Forward Iteration Python Replica)")
    plt.xlabel("Bar Index")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# === Backtesting with Swing-based SL ===
def backtest_strategy_swing_sl(df, results, sl_buffer, tp_multiplier):
    trades = []

    # Long trades (trigger up)
    for idx, mark_idx in zip(results["trigger_candles_up"], results["mark_candles_up"]):
        entry_price = df.loc[idx, "close"]
        sl_price = df.loc[mark_idx, "low"]  # last swing low
        tp_price = entry_price + tp_multiplier * (entry_price - sl_price)

        trade_outcome = None
        for j in range(idx + 1, len(df)):
            high = df.loc[j, "high"]
            low = df.loc[j, "low"]

            if low < sl_price - sl_buffer:  # stop-loss hit
                trade_outcome = sl_price - entry_price
                break
            elif high >= tp_price:  # take-profit hit
                trade_outcome = tp_price - entry_price
                break

        if trade_outcome is None:
            trade_outcome = df.loc[len(df) - 1, "close"] - entry_price

        trades.append(trade_outcome)

    # Short trades (trigger down)
    for idx, mark_idx in zip(results["trigger_candles_down"], results["mark_candles_down"]):
        entry_price = df.loc[idx, "close"]
        sl_price = df.loc[mark_idx, "high"]  # last swing high
        tp_price = entry_price - tp_multiplier * (sl_price - entry_price)

        trade_outcome = None
        for j in range(idx + 1, len(df)):
            high = df.loc[j, "high"]
            low = df.loc[j, "low"]

            if high > sl_price + sl_buffer:  # stop-loss hit
                trade_outcome = entry_price - sl_price
                break
            elif low <= tp_price:  # take-profit hit
                trade_outcome = entry_price - tp_price
                break

        if trade_outcome is None:
            trade_outcome = entry_price - df.loc[len(df) - 1, "close"]

        trades.append(trade_outcome)

    # Evaluation
    trades = np.array(trades)
    total_trades = len(trades)
    wins = np.sum(trades > 0)
    losses = np.sum(trades <= 0)
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0
    total_pnl = trades.sum()
    avg_pnl = total_pnl / total_trades if total_trades > 0 else 0

    print("--------------------------------------")
    print(f"Total Trades: {total_trades}")
    print(f"Wins: {wins}, Losses: {losses}, Win Rate: {win_rate:.2f}%")
    print(f"Total P&L: {round(total_pnl, 2)}")
    print(f"P&L per Trade: {round(avg_pnl, 2)}")
    print(f"Risk-Reward Ratio: {tp_multiplier}:1")

    return trades




# === Run Backtest ===
def main():
    u_wick_threshold = 4
    l_wick_threshold = 4
    buffer_window = 4
    wick_diff = 3.5
    st = 30
    stt = 15

    df = fetch_nifty_data(interval=5, days_back=90)
    results = find_marked_candles(df, u_wick_threshold, l_wick_threshold, buffer_window, wick_diff, st, stt)
    print("Analysis Result over past 90 days @ 5-min Nifty 50 spot, parameters:")
    print(f"Upper Wick Threshold: {u_wick_threshold}, Lower Wick Threshold: {l_wick_threshold}")
    print(f"Buffer Window: {buffer_window}, Wick Difference: {wick_diff}, Swing Thresholds: {st}, {stt}")
    
    print(f"Mark Up: {len(results['mark_candles_up'])}, Trigger Up: {len(results['trigger_candles_up'])}")
    print(f"Mark Down: {len(results['mark_candles_down'])}, Trigger Down: {len(results['trigger_candles_down'])}")

    # plot_candlestick_with_marks(df, results)

    trades = backtest_strategy_swing_sl(df, results, sl_buffer=20, tp_multiplier=3.5)

if __name__ == "__main__":
    main()



# ==============================
# FETCH NIFTY FUTURE DATA
# ==============================
# future_ticker = "NIFTY25OCTFUT"
# future_url = f"https://groww.in/v1/api/stocks_fo_data/v4/charting_service/chart/exchange/NSE/segment/FNO/{future_ticker}?endTimeInMillis={end_time_ms}&intervalInMinutes={interval}&startTimeInMillis={start_time_ms}"
# future_resp = requests.get(future_url)
# future_data = future_resp.json()["candles"]
# future_df = pd.DataFrame(future_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
# future_df["datetime"] = pd.to_datetime(future_df["timestamp"], unit="s").dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
# future_df = future_df[["datetime", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
