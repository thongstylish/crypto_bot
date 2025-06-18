import ccxt
import pandas as pd
import time
import joblib
import requests
from datetime import datetime
from pushbullet import Pushbullet
import re
import os
from ml_utils import load_symbols, add_indicators, load_model

PB_TOKEN = "o.6aa3gHCBx7xmf3Ws1IgmzIrI39UqbmoT"
FEATURES = ['rsi', 'macd', 'macd_signal', 'ema12', 'ema26']
MODEL_DIR = "models"

def fetch_data(symbol):
    binance = ccxt.binance({'timeout': 30000, 'enableRateLimit': True})
    ohlcv = binance.fetch_ohlcv(symbol, '4h', limit=100)
    df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    return df

def calculate_levels(df, window=3, num_levels=3, min_distance=30):
    highs, lows = df['high'], df['low']
    swing_highs, swing_lows = [], []
    for i in range(window, len(df) - window):
        if highs.iloc[i] == max(highs.iloc[i - window:i + window + 1]):
            swing_highs.append(highs.iloc[i])
        if lows.iloc[i] == min(lows.iloc[i - window:i + window + 1]):
            swing_lows.append(lows.iloc[i])
    def filter_levels(levels):
        filtered = []
        for lvl in sorted(set(levels)):
            if all(abs(lvl - x) > min_distance for x in filtered):
                filtered.append(lvl)
        return filtered
    return filter_levels(swing_lows)[:num_levels], filter_levels(swing_highs)[-num_levels:]

def analyze_entry(symbol, entry_price, entry_type, df):
    supports, resistances = calculate_levels(df)
    nearest_support = max([s for s in supports if s < entry_price], default=entry_price - 100)
    nearest_resistance = min([r for r in resistances if r > entry_price], default=entry_price + 150)
    stop_loss, take_profit = (nearest_support, nearest_resistance) if entry_type == 'long' else (nearest_resistance, nearest_support)
    trend_msg = f"=== PHÂN TÍCH LỆNH '{entry_type.upper()} {entry_price}' ({symbol}) ===\n"
    trend_msg += f"🕒 {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    trend_msg += f"💵 Giá vào lệnh: {entry_price}\n"
    trend_msg += f"🔻 Cắt lỗ: {stop_loss:.2f}\n"
    trend_msg += f"🔺 Chốt lời: {take_profit:.2f}\n"
    trend_msg += f"📉 Hỗ trợ: {', '.join([f'{s:.2f}' for s in supports])}\n"
    trend_msg += f"📈 Kháng cự: {', '.join([f'{r:.2f}' for r in resistances])}"
    return trend_msg

def listen_pushbullet():
    pb = Pushbullet(PB_TOKEN)
    last_push_id = None
    symbols = load_symbols()
    data_map = {}
    for symbol in symbols:
        df = fetch_data(symbol)
        df = add_indicators(df)
        data_map[symbol.lower()] = df
    print("🎧 Listening to Pushbullet commands...")
    while True:
        pushes = pb.get_pushes(limit=1)
        if pushes:
            push = pushes[0]
            if push['iden'] != last_push_id:
                last_push_id = push['iden']
                text = push.get("body", "").lower().strip()
                match = re.match(r'^(long|short)\s+(\d+(\.\d+)?)\s+([a-z]{3,5}/[a-z]{3,5})$', text)
                if match:
                    entry_type = match.group(1)
                    entry_price = float(match.group(2))
                    symbol = match.group(4).upper()
                    df = data_map.get(symbol.lower())
                    if df is not None:
                        msg = analyze_entry(symbol, entry_price, entry_type, df)
                        pb.push_note("Phân tích lệnh", msg)
                    else:
                        pb.push_note("Lỗi", f"Không tìm thấy dữ liệu cho {symbol}")
        time.sleep(10)

if __name__ == "__main__":
    listen_pushbullet()