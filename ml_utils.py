import os
import ta
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from pushbullet import Pushbullet

IMAGE_DIR = "images"
PB_TOKEN = "o.6aa3gHCBx7xmf3Ws1IgmzIrI39UqbmoT"
pb = Pushbullet(PB_TOKEN)

SYMBOL_TO_CHANNEL = {
    "ETH/USDT": "eth_channel",
    "BTC/USDT": "btc_channel",
    "XLM/USDT": "xlm_channel",
    "NEXO/USDT": "nexo_channel",
    "WLD/USDT": "wld_channel"
}


FEATURES = ['rsi', 'macd', 'macd_signal', 'ema12', 'ema26']

os.makedirs(IMAGE_DIR, exist_ok=True)

def load_symbols(file_path="symbols.txt"):
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def prepare_features(df):
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['ema12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
    df['ema26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
    return df.dropna()


def prepare_labels(df):
    df['target'] = 0
    for i in range(1, len(df)):
        pct = (df.iloc[i]['close'] - df.iloc[i - 1]['close']) / df.iloc[i - 1]['close']
        if pct > 0.01:
            df.loc[df.index[i], 'target'] = 1
        elif pct < -0.01:
            df.loc[df.index[i], 'target'] = -1
    return df['target']


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
        return filtered[:num_levels]

    return filter_levels(swing_lows), filter_levels(swing_highs)

def add_indicators(df):
    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['ema12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
    df['ema26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
    return df

def plot_chart(df, supports, resistances, symbol):
    import mplfinance as mpf

    df_plot = df.set_index("time")

    apds = [
        mpf.make_addplot(df["ema12"], color="blue"),
        mpf.make_addplot(df["ema26"], color="orange"),
        mpf.make_addplot(df["rsi"], panel=1, color="purple", ylabel="RSI"),
        mpf.make_addplot(df["macd"], panel=2, color="green", ylabel="MACD"),
        mpf.make_addplot(df["macd_signal"], panel=2, color="red")
    ]

    fig_filename = symbol.replace("/", "_") + ".png"
    fig_path = os.path.join(IMAGE_DIR, fig_filename)

    mpf.plot(
        df_plot,
        type="candle",
        style="charles",
        addplot=apds,
        title=symbol,
        ylabel="Giá",
        volume=True,
        savefig=dict(fname=fig_path, dpi=150, bbox_inches='tight')
    )

    return fig_path

def load_model(symbol, model_dir="models"):
    model_path = os.path.join(model_dir, f"{symbol.replace('/', '_')}_model.pkl")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

def notify_pushbullet(title, message, image_path=None, symbol=None):
    pb = Pushbullet(PB_TOKEN)

    upload_data = None
    if image_path and os.path.exists(image_path):
        file_name = os.path.basename(image_path)
        file_type = "image/png"
        with open(image_path, "rb") as f:
            # GỌI upload_file ĐÚNG THỨ TỰ ĐỐI SỐ
            upload_data = pb.upload_file(f, file_name, file_type)

    if symbol and symbol in SYMBOL_TO_CHANNEL:
        channel_tag = SYMBOL_TO_CHANNEL[symbol]
        channel = next((c for c in pb.channels if c.channel_tag == channel_tag), None)
        if channel is None:
            print(f"[ERROR] Không tìm thấy channel: {channel_tag}")
            return
        if upload_data:
            pb.push_file(
                file_url=upload_data["file_url"],
                file_name=upload_data["file_name"],
                file_type=upload_data["file_type"],
                body=message,
                channel=channel
            )
        else:
            pb.push_note(title, message, channel=channel)
    else:
        if upload_data:
            pb.push_file(
                file_url=upload_data["file_url"],
                file_name=upload_data["file_name"],
                file_type=upload_data["file_type"],
                body=message
            )
        else:
            pb.push_note(title, message)