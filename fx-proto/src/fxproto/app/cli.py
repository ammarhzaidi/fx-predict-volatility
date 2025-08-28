from __future__ import annotations
import typer
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

from fxproto.config.loader import get_config, resolved_outputs_dir
from fxproto.data.preprocess import load_or_fetch, train_test_split_by_dates, scale_train_apply_test
from fxproto.data.features import basic_features, make_supervised_windows
from fxproto.models.lstm import TinyLSTM, train_model
from fxproto.models.infer import predict_series, add_predictions_to_frame

app = typer.Typer(add_completion=False)

@app.command()
def forecast():
    """Fetch → preprocess → features → train tiny LSTM → save CSV + chart."""
    cfg = get_config()
    pair = cfg.settings.pair
    interval = cfg.data.interval
    dates = cfg.settings.dates
    out_dir = resolved_outputs_dir()
    (out_dir / "charts").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)

    # 1) Data
    df = load_or_fetch(pair, dates.train_start, dates.test_end, interval)
    df_feat = basic_features(df)

    # 2) Train/Test split + scaling
    train, test = train_test_split_by_dates(df_feat)
    feature_cols = ["ret_1", "ma_5", "vol_10"]
    target_col = "Close"
    train, test, scaler = scale_train_apply_test(train, test, cols=feature_cols)

    # 3) Windows
    lookback = 30
    horizon = cfg.settings.horizon
    Xtr, ytr = make_supervised_windows(train, feature_cols, target_col, lookback, horizon)
    Xte, yte = make_supervised_windows(test, feature_cols, target_col, lookback, horizon)

    # 4) Train
    model = TinyLSTM(n_features=len(feature_cols))
    model = train_model(model, Xtr, ytr, epochs=15, lr=1e-3)

    # 5) Predict
    preds = predict_series(model, Xte)
    # figure out alignment index offset relative to the full df_feat
    # first test window starts at index: len(train) + lookback
    start_idx = len(train) + lookback
    report = add_predictions_to_frame(df_feat, start_idx, preds, target_col, horizon)

    # 6) Save CSV + Plot
    csv_path = out_dir / "reports" / f"{pair}_forecast_h{horizon}.csv"
    report.to_csv(csv_path, index=True)

    plt.figure(figsize=(11,5))
    plt.plot(df_feat.index, df_feat["Close"], label="Close", alpha=0.6)
    plt.plot(report.index, report["pred"], label="LSTM forecast", linewidth=2)
    plt.title(f"{pair} forecast (h={horizon})")
    plt.legend(); plt.grid(True)
    png_path = out_dir / "charts" / f"{pair}_forecast_h{horizon}.png"
    plt.savefig(png_path, dpi=140, bbox_inches="tight")
    print(f"[OK] Wrote: {csv_path}")
    print(f"[OK] Wrote: {png_path}")

if __name__ == "__main__":
    app()
