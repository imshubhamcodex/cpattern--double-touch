import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tabulate import tabulate
import requests
from datetime import datetime, timedelta
import pytz


# ==============================
# Load CSV Data
# ==============================
def load_nifty_csv(file_path):
    """
    Reads NIFTY CSV data, parses datetime, and formats it for backtesting.
    Handles both timezone-aware and naive datetimes.
    """
    df = pd.read_csv(file_path)

    # Parse datetime
    df["datetime"] = pd.to_datetime(df["date"], utc=True)

    # Convert to IST (Asia/Kolkata)
    df["datetime"] = df["datetime"].dt.tz_convert('Asia/Kolkata')

    # Keep only required columns
    df = df[["datetime", "open", "high", "low", "close"]].copy()

    # Reset index
    df = df.reset_index(drop=True)

    return df

# ==============================
# Data Fetching
# ==============================
def fetch_nifty_data(interval=5, days_back=30):
    """Fetch NIFTY 50 historical data from Groww API."""
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
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")\
                      .dt.tz_localize('UTC')\
                      .dt.tz_convert('Asia/Kolkata')
    return df.reset_index(drop=True)

# ==============================
# Swing Detection
# ==============================
def detect_swing(df, start_idx, end_idx, swing_threshold):
    """Detects if a swing exists between start_idx and end_idx."""
    if end_idx - start_idx + 1 <= 0:
        return False

    swing_high = max(df.loc[start_idx:end_idx, "high"])
    swing_low = min(df.loc[start_idx:end_idx, "low"])
    return (swing_high - swing_low) > swing_threshold

# ==============================
# Candle Detection
# ==============================
def find_marked_candles(df, u_wick_threshold, l_wick_threshold, buffer_window, wick_diff, st, stt, sl_buffer, tp_multiplier):
    """Identify marked and trigger candles."""
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

    mark_up, test_up, trigger_up, mark_up_dup = [], [], [], []
    mark_dn, test_dn, trigger_dn, mark_dn_dup = [], [], [], []

    for i in range(len(df)):
        # --- Mark Up Candles ---
        if df.loc[i, "upper_wick"] >= u_wick_threshold:
            for j in range(i + 1, len(df)):
                diff = df.loc[j, "high"] - df.loc[i, "high"]
                if diff > 0:
                    break
                if abs(diff) <= wick_diff and j >= i + buffer_window:
                    if df.loc[j, "high"] - max(df.loc[j, "open"], df.loc[j, "close"]) >= u_wick_threshold:
                        if detect_swing(df, i, j, st):
                            mark_up.append(j)
                            test_up.append(i)

        # --- Mark Down Candles ---
        if df.loc[i, "lower_wick"] >= l_wick_threshold:
            for j in range(i + 1, len(df)):
                diff = df.loc[i, "low"] - df.loc[j, "low"]
                if diff > 0:
                    break
                if abs(diff) <= wick_diff and j >= i + buffer_window:
                    if min(df.loc[j, "open"], df.loc[j, "close"]) - df.loc[j, "low"] >= l_wick_threshold:
                        if detect_swing(df, i, j, st):
                            mark_dn.append(j)
                            test_dn.append(i)

    # --- Trigger Candles ---
    def find_trigger(mark_list, buffer_window, stt, direction="up"):
        trigger, mark_dup = [], []
        for idx in mark_list:
            for j in range(idx + 1, len(df)):
                if direction == "up":
                    diff = df.loc[j, "high"] - df.loc[idx, "high"]
                else:
                    diff = df.loc[idx, "low"] - df.loc[j, "low"]

                if diff > 0 and j < idx + buffer_window:
                    break
                if diff > 0 and j >= idx + buffer_window:
                    if detect_swing(df, idx, j, stt):
                        trigger.append(j)
                        mark_dup.append(idx)
                    break
        return trigger, mark_dup

    trigger_up, mark_up_dup = find_trigger(mark_up, buffer_window, stt, "up")
    trigger_dn, mark_dn_dup = find_trigger(mark_dn, buffer_window, stt, "down")

    return {
        "mark_candles_up": mark_up_dup,
        "test_candles_origin_up": test_up,
        "trigger_candles_up": trigger_up,
        "mark_candles_down": mark_dn_dup,
        "test_candles_origin_down": test_dn,
        "trigger_candles_down": trigger_dn
    }

# ==============================
# Backtesting
# ==============================
def backtest_strategy_swing_sl(df, results, sl_buffer, tp_multiplier):
    trades = []

    # --- Long Trades ---
    for idx, mark_idx in zip(results["trigger_candles_up"], results["mark_candles_up"]):
        entry, sl = df.loc[idx, "close"], df.loc[mark_idx, "low"]
        tp = entry + tp_multiplier * (entry - sl)
        trade_outcome = None
        for j in range(idx + 1, len(df)):
            if df.loc[j, "low"] < sl - sl_buffer:
                trade_outcome = sl - entry
                break
            elif df.loc[j, "high"] >= tp:
                trade_outcome = tp - entry
                break
        trades.append(trade_outcome if trade_outcome is not None else df.loc[len(df)-1, "close"] - entry)

    # --- Short Trades ---
    for idx, mark_idx in zip(results["trigger_candles_down"], results["mark_candles_down"]):
        entry, sl = df.loc[idx, "close"], df.loc[mark_idx, "high"]
        tp = entry - tp_multiplier * (sl - entry)
        trade_outcome = None
        for j in range(idx + 1, len(df)):
            if df.loc[j, "high"] > sl + sl_buffer:
                trade_outcome = entry - sl
                break
            elif df.loc[j, "low"] <= tp:
                trade_outcome = entry - tp
                break
        trades.append(trade_outcome if trade_outcome is not None else entry - df.loc[len(df)-1, "close"])

    trades = np.array(trades)
    total_trades, wins, losses = len(trades), np.sum(trades>0), np.sum(trades<=0)
    win_rate = round(wins / total_trades * 100, 2) if total_trades else 0
    total_pnl = round(trades.sum(), 2)
    avg_pnl = round(total_pnl / total_trades, 2) if total_trades else 0

    summary_df = pd.DataFrame([{
        "Total Trades": total_trades,
        "Wins": wins,
        "Losses": losses,
        "Win Rate (%)": win_rate,
        "Total P&L": total_pnl,
        "P&L per Trade": avg_pnl,
        "Risk-Reward Ratio": f"{tp_multiplier}:1"
    }])

    print("\nBacktest Summary:")
    print(tabulate(summary_df, headers='keys', tablefmt='fancy_grid', showindex=False))

    return trades

# ==============================
# Display Mark ↔ Trigger Candle Table
# ==============================
def print_marked_trigger_table(df, results):
    data = []
    for direction, mark_list, trigger_list in [("Up", results["mark_candles_up"], results["trigger_candles_up"]),
                                               ("Down", results["mark_candles_down"], results["trigger_candles_down"])]:
        for mark_idx, trigger_idx in zip(mark_list, trigger_list):
            data.append({
                "Type": direction,
                "Mark Candle Datetime": df.loc[mark_idx, "datetime"],
                "Trigger Candle Datetime": df.loc[trigger_idx, "datetime"],
                "Mark Candle Index": mark_idx,
                "Trigger Candle Index": trigger_idx
            })

    table_df = pd.DataFrame(data).sort_values("Mark Candle Datetime").reset_index(drop=True)
    print("\nMarked Candle ↔ Trigger Candle Table:")
    print(tabulate(table_df, headers='keys', tablefmt='fancy_grid'))


# # === Visualization ===
# def plot_candlestick_with_marks(df, results):
#     plt.figure(figsize=(15, 8))
#     for i in range(len(df)):
#         color = 'green' if df.loc[i, "close"] >= df.loc[i, "open"] else 'red'
#         plt.plot([i, i], [df.loc[i, "low"], df.loc[i, "high"]], color='black', linewidth=1)
#         plt.plot([i, i], [df.loc[i, "open"], df.loc[i, "close"]], color=color, linewidth=3)

#     # Plot markers
#     if results["mark_candles_up"]:
#         plt.scatter(results["mark_candles_up"], df.loc[results["mark_candles_up"], "low"] - 5,
#                     marker="^", color="orange", s=80, label="Mark Up")
#     if results["trigger_candles_up"]:
#         plt.scatter(results["trigger_candles_up"], df.loc[results["trigger_candles_up"], "low"] - 8,
#                     marker="^", color="green", s=80, label="Trigger Up")
#     if results["mark_candles_down"]:
#         plt.scatter(results["mark_candles_down"], df.loc[results["mark_candles_down"], "high"] + 5,
#                     marker="v", color="orange", s=80, label="Mark Down")
#     if results["trigger_candles_down"]:
#         plt.scatter(results["trigger_candles_down"], df.loc[results["trigger_candles_down"], "high"] + 8,
#                     marker="v", color="red", s=80, label="Trigger Down")

#     plt.title("NIFTY Marked Candles (Forward Iteration Python Replica)")
#     plt.xlabel("Bar Index")
#     plt.ylabel("Price")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

def plot_candlestick_with_marks_plotly(df, results):
    fig = go.Figure()

    # Add candlestick
    fig.add_trace(go.Candlestick(
        x=df["datetime"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        increasing_line_color='green',
        decreasing_line_color='red',
        name='Price'
    ))

    # Add Mark Up candles
    if results["mark_candles_up"]:
        fig.add_trace(go.Scatter(
            x=df.loc[results["mark_candles_up"], "datetime"],
            y=df.loc[results["mark_candles_up"], "low"] - 5,
            mode='markers',
            marker=dict(symbol='triangle-up', color='orange', size=12),
            name='Mark Up'
        ))

    # Add Trigger Up candles
    if results["trigger_candles_up"]:
        fig.add_trace(go.Scatter(
            x=df.loc[results["trigger_candles_up"], "datetime"],
            y=df.loc[results["trigger_candles_up"], "low"] - 8,
            mode='markers',
            marker=dict(symbol='triangle-up', color='green', size=12),
            name='Trigger Up'
        ))

    # Add Mark Down candles
    if results["mark_candles_down"]:
        fig.add_trace(go.Scatter(
            x=df.loc[results["mark_candles_down"], "datetime"],
            y=df.loc[results["mark_candles_down"], "high"] + 5,
            mode='markers',
            marker=dict(symbol='triangle-down', color='orange', size=12),
            name='Mark Down'
        ))

    # Add Trigger Down candles
    if results["trigger_candles_down"]:
        fig.add_trace(go.Scatter(
            x=df.loc[results["trigger_candles_down"], "datetime"],
            y=df.loc[results["trigger_candles_down"], "high"] + 8,
            mode='markers',
            marker=dict(symbol='triangle-down', color='red', size=12),
            name='Trigger Down'
        ))

    # Layout customization
    fig.update_layout(
        title='NIFTY Marked Candles (Interactive)',
        xaxis_title='Datetime',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        height=600
    )

    fig.show()



# ==============================
# Main
# ==============================
def main():
    # Parameters
    params = {
        "u_wick_threshold": 4,
        "l_wick_threshold": 4,
        "buffer_window": 4,
        "wick_diff": 3.5,
        "st": 30,
        "stt": 15,
        "sl_buffer": 20,
        "tp_multiplier": 3.5
    }

    # df = fetch_nifty_data(interval=5, days_back=90)
    load_nifty_csv_path = "NIFTY50-5.csv"  
    df = load_nifty_csv(load_nifty_csv_path)

    results = find_marked_candles(df, **params)

    # Analysis Parameters Table
    param_table = [[k.replace("_"," ").title(), v] for k,v in params.items()]
    print("\nAnalysis Parameters (Past 90 days, 5-min Nifty 50 spot):")
    print(tabulate(param_table, headers=["Parameter","Value"], tablefmt="fancy_grid"))

    # Marked Candles Summary
    print("\nMarked Candles Summary:")
    print(tabulate([
        ["Trigger Up", len(results['trigger_candles_up'])],
        ["Trigger Down", len(results['trigger_candles_down'])]
    ], headers=["Type","Count"], tablefmt="fancy_grid"))

    # Backtest
    backtest_strategy_swing_sl(df, results, sl_buffer=params["sl_buffer"], tp_multiplier=params["tp_multiplier"])

    # Detailed Mark ↔ Trigger Table
    print_marked_trigger_table(df, results)


    # plot_candlestick_with_marks(df, results)
    plot_candlestick_with_marks_plotly(df, results)

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

# ==============================
