# scripts/test_fetch_2.py

import matplotlib.pyplot as plt
from fxproto.data import fetch_ohlcv

def main():
    df = fetch_ohlcv("EURUSD", start="2024-08-26", end="2025-08-26", interval="1d", save_csv=True)
    print(df.head())

    plt.figure(figsize=(10,5))
    plt.plot(df.index, df["Close"], label="EUR/USD Close")
    plt.title("EUR/USD - Daily Closing Price")
    plt.xlabel("Date"); plt.ylabel("Price"); plt.grid(True); plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

