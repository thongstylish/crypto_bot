import ccxt
import pandas as pd
import ta
import joblib
import time
import os
from sklearn.ensemble import RandomForestClassifier
from ml_utils import prepare_features, prepare_labels, calculate_levels, plot_chart, notify_pushbullet

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def get_symbols(filename='symbols.txt'):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def fetch_data(symbol):
    binance = ccxt.binance()
    ohlcv = binance.fetch_ohlcv(symbol, '4h', limit=100)
    df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df

def process_symbol(symbol):
    df = fetch_data(symbol)
    df = prepare_features(df)
    X = df[["rsi", "macd", "macd_signal", "ema12", "ema26"]].dropna()
    y = prepare_labels(df)

    model_path = os.path.join(MODEL_DIR, f"{symbol.replace('/', '_')}_model.pkl")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        joblib.dump(model, model_path)

    prediction = model.predict(X.iloc[[-1]])[0]
    signal_map = {-1: "BÁN", 0: "GIỮ", 1: "MUA"}
    signal = signal_map.get(prediction, "KHÔNG RÕ")

    supports, resistances = calculate_levels(df)
    image_path = plot_chart(df, supports, resistances, symbol)

    message = f"=== TÍN HIỆU {symbol} ===\n🕓 {df['time'].iloc[-1]}\n"
    message += f"🔹 RSI: {df['rsi'].iloc[-1]:.2f}\n"
    message += f"🔹 MACD: {df['macd'].iloc[-1]:.2f}, Signal: {df['macd_signal'].iloc[-1]:.2f}\n"
    message += f"🔹 EMA12: {df['ema12'].iloc[-1]:.2f}, EMA26: {df['ema26'].iloc[-1]:.2f}\n"
    message += f"📉 Hỗ trợ: {', '.join([f'{s:.2f}' for s in supports])}\n"
    message += f"📈 Kháng cự: {', '.join([f'{r:.2f}' for r in resistances])}\n"
    message += f"💡 Dự đoán: {signal}"

    notify_pushbullet(f"Tín hiệu {symbol}", message, image_path, symbol)

if __name__ == "__main__":
    while True:
        symbols = get_symbols()
        for symbol in symbols:
            try:
                process_symbol(symbol)
            except Exception as e:
                print(f"[ERROR] {symbol}: {e}")
        time.sleep(600)